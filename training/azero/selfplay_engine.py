"""Game-parallel self-play with optional leaf-parallel (virtual loss) MCTS.

Everything routes through a single ``evaluate`` callable that maps a batch of
encoded positions ``(N, C, H, W)`` to ``(policies (N, P), values (N,))``. The
coordinator keeps many games in flight at once and gathers every pending leaf
from every game into one ``evaluate`` call, so whoever owns the model (a local
network or the remote inference server) always sees a large batch instead of
batch-1 calls.

Two independent batch multipliers, both lossless at their defaults:
  * ``selfplay_concurrent_games`` -- games run concurrently per coordinator;
    each contributes its current leaf to the shared batch. Zero quality impact
    (each game's search is unchanged).
  * ``mcts_leaf_batch`` -- leaves gathered per game per round. ``1`` is exactly
    sequential MCTS; ``>1`` uses virtual loss to fan out within one tree, a
    small speed/quality trade-off (configurable, off by default).

Tree semantics (PUCT selection, Dirichlet root noise, sign-alternating backup,
visit-count move selection) mirror :mod:`azero.mcts`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, List, Optional, Tuple

import numpy as np

from goengine import Color, Game, Move

from azero.config import Config
from azero.features import HistoryTracker, encode, index_to_move, move_to_index
from azero.mcts import Node
from azero.replay import Sample

# evaluate(planes (N,C,H,W) float32) -> (policies (N,P) float, values (N,) float)
EvaluateFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclasses.dataclass
class SelfPlayStats:
    """Throughput counters for one self-play run."""

    games: int = 0
    positions: int = 0  # network evaluations performed
    forward_calls: int = 0  # evaluate() invocations

    @property
    def avg_batch(self) -> float:
        return self.positions / self.forward_calls if self.forward_calls else 0.0


# -- tree helpers (policy supplied externally, not by a bound net) -------------


def _clone_tracker(tracker: HistoryTracker) -> HistoryTracker:
    clone = HistoryTracker(tracker.history)
    for board in reversed(tracker.boards()):
        clone.push(board)
    return clone


def _select_child(node: Node, c_puct: float) -> Tuple[int, Node]:
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


# -- one game's stepped, batchable search -------------------------------------


class _GameSearch:
    """A single self-play game whose MCTS yields leaves for external batching."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.game = Game(config.board_size, config.komi)
        self.tracker = HistoryTracker(config.history_planes, self.game.board)
        self.move_number = 0
        self.done = False
        self.samples: List[Sample] = []
        self._sample_pending: List[Tuple[np.ndarray, np.ndarray, Color]] = []
        self._start_move()

    def _start_move(self) -> None:
        self.root = Node(prior=1.0)
        self.root_expanded = False
        self.root_planes: Optional[np.ndarray] = None
        self.descents = 0
        self._root_request = False
        self._leaves: List[Tuple[List[Node], Game]] = []
        self._pending_ids: set = set()

    # -- batched stepping --------------------------------------------------

    def collect(self) -> List[np.ndarray]:
        """Planes needing evaluation this round: the root, then leaf batches."""
        if self.done:
            return []
        cfg = self.config
        if not self.root_expanded:
            if self.root_planes is None:
                self.root_planes = encode(
                    self.game.board, self.tracker,
                    self.game.current_player, cfg.history_planes,
                )
            self._root_request = True
            return [self.root_planes]

        self._root_request = False
        self._leaves = []
        self._pending_ids = set()
        planes: List[np.ndarray] = []
        budget = cfg.simulations - self.descents
        if budget <= 0:
            return []
        want = min(max(1, cfg.mcts_leaf_batch), budget)
        guard = 0
        while len(planes) < want and self.descents < cfg.simulations and guard < want * 4:
            guard += 1
            result = self._descend()
            if result is None:  # terminal leaf, already backed up + counted
                continue
            if result[0] == "collision":  # leaf already queued this round
                break
            _, path, leaf_game, leaf_planes = result
            self._apply_virtual_loss(path)
            self._leaves.append((path, leaf_game))
            self._pending_ids.add(id(path[-1]))
            planes.append(leaf_planes)
            self.descents += 1
        return planes

    def _descend(self):
        cfg = self.config
        size = cfg.board_size
        node = self.root
        sim = self.game.copy()
        sim_tracker = _clone_tracker(self.tracker)
        path = [node]
        while node.expanded and not sim.is_over:
            index, node = _select_child(node, cfg.c_puct)
            sim.play(index_to_move(index, size))
            sim_tracker.push(sim.board)
            path.append(node)
        if sim.is_over:
            self._backup(path, _terminal_value(sim))
            self.descents += 1
            return None
        if id(node) in self._pending_ids:
            return ("collision",)
        leaf_planes = encode(
            sim.board, sim_tracker, sim.current_player, cfg.history_planes
        )
        return ("leaf", path, sim, leaf_planes)

    def consume(self, policies: np.ndarray, values: np.ndarray) -> None:
        """Apply the evaluations requested by the last ``collect()``."""
        if self._root_request:
            _expand_node(self.root, self.game, policies[0], self.config)
            self._add_dirichlet_noise(self.root)
            self.root_expanded = True
            self._root_request = False
            return
        for (path, leaf_game), policy, value in zip(self._leaves, policies, values):
            self._remove_virtual_loss(path)
            _expand_node(path[-1], leaf_game, policy, self.config)
            self._backup(path, float(value))
        self._leaves = []
        self._pending_ids = set()

    def move_ready(self) -> bool:
        return self.root_expanded and self.descents >= self.config.simulations

    # -- move completion ---------------------------------------------------

    def commit_move(self) -> None:
        cfg = self.config
        visits = np.zeros(cfg.policy_size, dtype=np.float32)
        for index, child in self.root.children.items():
            visits[index] = child.visit_count
        total = visits.sum()
        if total > 0:
            visits /= total

        player = self.game.current_player
        assert self.root_planes is not None
        self._sample_pending.append((self.root_planes, visits, player))

        move = self._select_move(visits)
        self.game.play(move)
        self.tracker.push(self.game.board)
        self.move_number += 1
        self._start_move()

        if self.game.is_over or self.move_number >= cfg.move_cap:
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
        for planes, policy, player in self._sample_pending:
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

    def _apply_virtual_loss(self, path: List[Node]) -> None:
        vl = self.config.virtual_loss
        if vl <= 0:
            return
        value = -1.0  # pessimistic: the descended path "lost"
        for node in reversed(path):
            node.visit_count += vl
            node.value_sum += vl * value
            value = -value

    def _remove_virtual_loss(self, path: List[Node]) -> None:
        vl = self.config.virtual_loss
        if vl <= 0:
            return
        value = -1.0
        for node in reversed(path):
            node.visit_count -= vl
            node.value_sum -= vl * value
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


def run_self_play(
    evaluate: EvaluateFn,
    config: Config,
    total_games: int,
    concurrent: int,
    stats: Optional[SelfPlayStats] = None,
) -> List[Sample]:
    """Play ``total_games`` games, keeping ``concurrent`` in flight at once."""
    concurrent = max(1, min(concurrent, total_games))
    samples: List[Sample] = []
    started = 0
    active: List[_GameSearch] = []

    while started < total_games or active:
        while len(active) < concurrent and started < total_games:
            active.append(_GameSearch(config))
            started += 1
        if not active:
            break

        # Gather every active game's pending positions into one batch.
        batch: List[np.ndarray] = []
        slices: List[Tuple[_GameSearch, int, int]] = []
        for gs in active:
            planes = gs.collect()
            if planes:
                slices.append((gs, len(batch), len(planes)))
                batch.extend(planes)

        if batch:
            arr = np.stack(batch)
            policies, values = evaluate(arr)
            if stats is not None:
                stats.forward_calls += 1
                stats.positions += len(batch)
            for gs, start, count in slices:
                gs.consume(policies[start:start + count], values[start:start + count])

        # Commit finished moves; retire finished games.
        still: List[_GameSearch] = []
        for gs in active:
            if gs.move_ready():
                gs.commit_move()
            if gs.done:
                samples.extend(gs.samples)
                if stats is not None:
                    stats.games += 1
            else:
                still.append(gs)
        active = still

    return samples
