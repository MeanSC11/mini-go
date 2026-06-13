"""Vectorized self-play: many concurrent games, one batched GPU evaluation.

The multi-process self-play path evaluates one leaf at a time (batch size 1),
which leaves a GPU almost completely idle. Here a single process drives many
games at once and gathers the leaf to evaluate from every active game's search
tree into a single ``predict_many`` call. The GPU sees batches of roughly
``parallel_games`` positions instead of N separate batch-1 forward passes.

The tree semantics (PUCT selection, Dirichlet root noise, sign-alternating
backup, visit-count move selection) mirror :mod:`azero.mcts` exactly; only the
control flow is inverted so leaf evaluation can be batched across games.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from goengine import Color, Game, Move

from azero.config import Config
from azero.features import (
    HistoryTracker,
    encode,
    index_to_move,
    move_to_index,
)
from azero.mcts import Node
from azero.network import PolicyValueNet
from azero.replay import Sample

logger = logging.getLogger(__name__)


# -- tree helpers (shared shape with azero.mcts, policy supplied externally) ---


def _clone_tracker(tracker: HistoryTracker) -> HistoryTracker:
    clone = HistoryTracker(tracker.history)
    for board in reversed(tracker.boards()):
        clone.push(board)
    return clone


def _select_child(node: Node, c_puct: float) -> Tuple[int, Node]:
    """PUCT: argmax over Q(child from parent's view) + U."""
    sqrt_n = math.sqrt(node.visit_count)
    best_score = -float("inf")
    best: Tuple[int, Node] = next(iter(node.children.items()))
    for index, child in node.children.items():
        u = c_puct * child.prior * sqrt_n / (1 + child.visit_count)
        score = -child.q + u
        if score > best_score:
            best_score = score
            best = (index, child)
    return best


def _expand_node(node: Node, game: Game, policy: np.ndarray, config: Config) -> None:
    """Create child edges for ``game``'s legal moves from network priors."""
    legal = game.legal_moves()
    size = game.board_size
    priors = np.zeros(config.policy_size, dtype=np.float64)
    for move in legal:
        priors[move_to_index(move, size)] = float(policy[move_to_index(move, size)])
    total = priors.sum()
    if total <= 1e-9:
        for move in legal:
            priors[move_to_index(move, size)] = 1.0
        total = priors.sum()
    priors /= total
    for index in np.nonzero(priors)[0]:
        node.children[int(index)] = Node(prior=float(priors[index]))


def _terminal_value(game: Game) -> float:
    result = game.result
    assert result is not None
    if result.winner is game.current_player:
        return 1.0
    if result.winner is game.current_player.opponent:
        return -1.0
    return 0.0


def _winner(game: Game) -> Color:
    if game.result is not None:
        return game.result.winner
    black, white = game.score()
    if black > white:
        return Color.BLACK
    if white > black:
        return Color.WHITE
    return Color.EMPTY


# -- one game's stepped search ------------------------------------------------


class _GameSearch:
    """A single self-play game whose MCTS can be stepped one leaf at a time.

    Each move runs ``simulations`` simulations. ``request()`` returns the planes
    that need a network evaluation (or ``None`` when nothing is pending this
    round), and ``apply()`` feeds the result back. ``commit_move()`` finishes
    the move once all simulations are done.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.game = Game(config.board_size, config.komi)
        self.tracker = HistoryTracker(config.history_planes, self.game.board)
        self.move_number = 0
        self.done = False
        self.samples: List[Sample] = []
        self._pending: List[Tuple[np.ndarray, np.ndarray, Color]] = []
        self._reset_move()

    def _reset_move(self) -> None:
        self.root = Node(prior=1.0)
        self.root_expanded = False
        self.root_planes: Optional[np.ndarray] = None
        self.sims_done = 0
        self._leaf_path: Optional[List[Node]] = None
        self._leaf_game: Optional[Game] = None

    # -- batched stepping --------------------------------------------------

    def request(self) -> Optional[np.ndarray]:
        """Planes needing evaluation this round, or ``None`` if none."""
        size = self.config.board_size
        if not self.root_expanded:
            self.root_planes = encode(
                self.game.board, self.tracker,
                self.game.current_player, self.config.history_planes,
            )
            self._leaf_path = None  # root expansion handled in apply()
            return self.root_planes
        if self.sims_done >= self.config.simulations:
            return None
        # Run one simulation's selection down to a leaf.
        node = self.root
        sim = self.game.copy()
        sim_tracker = _clone_tracker(self.tracker)
        path = [node]
        while node.expanded and not sim.is_over:
            index, node = _select_child(node, self.config.c_puct)
            sim.play(index_to_move(index, size))
            sim_tracker.push(sim.board)
            path.append(node)
        if sim.is_over:
            # Terminal leaf: no network needed, backup immediately.
            self._backup(path, _terminal_value(sim))
            self.sims_done += 1
            return None
        self._leaf_path = path
        self._leaf_game = sim
        return encode(
            sim.board, sim_tracker, sim.current_player, self.config.history_planes
        )

    def apply(self, policy: np.ndarray, value: float) -> None:
        """Consume the evaluation requested by the last ``request()``."""
        if not self.root_expanded:
            _expand_node(self.root, self.game, policy, self.config)
            self._add_dirichlet_noise(self.root)
            self.root_expanded = True
            return
        assert self._leaf_path is not None and self._leaf_game is not None
        leaf = self._leaf_path[-1]
        _expand_node(leaf, self._leaf_game, policy, self.config)
        self._backup(self._leaf_path, float(value))
        self.sims_done += 1
        self._leaf_path = None
        self._leaf_game = None

    # -- move completion ---------------------------------------------------

    def commit_move(self) -> None:
        """Pick a move from visit counts, advance the game, reset the tree."""
        size = self.config.board_size
        visits = np.zeros(self.config.policy_size, dtype=np.float32)
        for index, child in self.root.children.items():
            visits[index] = child.visit_count
        total = visits.sum()
        if total > 0:
            visits /= total

        player = self.game.current_player
        assert self.root_planes is not None
        self._pending.append((self.root_planes, visits, player))

        move = self._select_move(visits)
        self.game.play(move)
        self.tracker.push(self.game.board)
        self.move_number += 1
        self._reset_move()

        if self.game.is_over or self.move_number >= self.config.move_cap:
            self._finalize()

    def _select_move(self, policy: np.ndarray) -> Move:
        if policy.sum() == 0:
            return Move.pass_turn()
        temperature = 1.0 if self.move_number < self.config.temperature_moves else 0.0
        if temperature <= 1e-6:
            index = int(policy.argmax())
        else:
            weighted = np.power(policy, 1.0 / temperature)
            weighted /= weighted.sum()
            index = int(np.random.choice(len(weighted), p=weighted))
        return index_to_move(index, self.config.board_size)

    def _finalize(self) -> None:
        winner = _winner(self.game)
        for planes, policy, player in self._pending:
            if winner is Color.EMPTY:
                z = 0.0
            else:
                z = 1.0 if winner is player else -1.0
            self.samples.append((planes, policy, z))
        self.done = True

    # -- internals ---------------------------------------------------------

    def _backup(self, path: List[Node], value: float) -> None:
        for ancestor in reversed(path):
            ancestor.visit_count += 1
            ancestor.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, root: Node) -> None:
        indices = list(root.children.keys())
        if not indices:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(indices))
        eps = self.config.dirichlet_eps
        for index, n in zip(indices, noise):
            child = root.children[index]
            child.prior = (1 - eps) * child.prior + eps * float(n)


# -- coordinator --------------------------------------------------------------


def generate_games_batched(
    net: PolicyValueNet, config: Config, total_games: int
) -> List[Sample]:
    """Play ``total_games`` games concurrently, batching all leaf evaluations.

    Keeps up to ``parallel_games`` games in flight; whenever one finishes a new
    one starts, so the batch stays near full until the final cohort.
    """
    parallel = config.selfplay_parallel_games or min(total_games, 128)
    parallel = max(1, min(parallel, total_games))
    simulations = config.simulations

    samples: List[Sample] = []
    started = 0
    active: List[_GameSearch] = []

    while started < total_games or active:
        while len(active) < parallel and started < total_games:
            active.append(_GameSearch(config))
            started += 1
        if not active:
            break

        # One move for every active game: root expansion round + N sim rounds.
        for _ in range(simulations + 1):
            requests: List[Tuple[_GameSearch, np.ndarray]] = []
            for gs in active:
                planes = gs.request()
                if planes is not None:
                    requests.append((gs, planes))
            if not requests:
                continue
            batch = np.stack([planes for _, planes in requests])
            policies, values = net.predict_many(batch)
            for (gs, _), policy, value in zip(requests, policies, values):
                gs.apply(policy, float(value))

        still_active: List[_GameSearch] = []
        for gs in active:
            gs.commit_move()
            if gs.done:
                samples.extend(gs.samples)
            else:
                still_active.append(gs)
        active = still_active

    return samples
