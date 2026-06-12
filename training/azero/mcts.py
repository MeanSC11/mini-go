"""PUCT Monte-Carlo Tree Search guided by a policy+value network.

Follows the AlphaZero paper: leaf evaluation by the network (no rollouts),
PUCT selection, Dirichlet noise at the root during self-play, and visit-count
move selection with an optional temperature.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from goengine import Game, Move

from azero.config import Config
from azero.features import HistoryTracker, encode, index_to_move, move_to_index
from azero.network import PolicyValueNet


class Node:
    """A search tree node; edges are policy indices."""

    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}

    @property
    def q(self) -> float:
        """Mean action value from the perspective of the player to move."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    """One search instance bound to a network and config."""

    def __init__(self, net: PolicyValueNet, config: Config) -> None:
        self.net = net
        self.config = config

    # -- public API -------------------------------------------------------

    def run(
        self,
        game: Game,
        tracker: Optional[HistoryTracker] = None,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Search from ``game`` and return ``(visit_distribution, root_value)``.

        ``visit_distribution`` is over policy indices (board points + pass)
        and sums to 1. ``root_value`` is the network's value estimate of the
        root for the side to move.
        """
        size = game.board_size
        root = Node(prior=1.0)
        root_value = self._expand(root, game, tracker)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.simulations):
            node = root
            sim = game.copy()
            sim_tracker = self._clone_tracker(tracker)
            path = [node]
            # Selection
            while node.expanded and not sim.is_over:
                index, node = self._select_child(node)
                move = index_to_move(index, size)
                sim.play(move)
                if sim_tracker is not None:
                    sim_tracker.push(sim.board)
                path.append(node)
            # Evaluation / expansion
            if sim.is_over:
                value = self._terminal_value(sim)
            else:
                value = self._expand(node, sim, sim_tracker)
            # Backup: ``value`` is from the perspective of the player to move
            # at the evaluated node; alternate sign up the path.
            for ancestor in reversed(path):
                ancestor.visit_count += 1
                ancestor.value_sum += value
                value = -value

        visits = np.zeros(self.config.policy_size, dtype=np.float32)
        for index, child in root.children.items():
            visits[index] = child.visit_count
        total = visits.sum()
        if total > 0:
            visits /= total
        return visits, root_value

    def choose_move(
        self,
        game: Game,
        tracker: Optional[HistoryTracker] = None,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> Tuple[Move, np.ndarray, float]:
        """Search and pick a move.

        ``temperature == 0`` plays the most-visited move; otherwise moves are
        sampled proportionally to ``visits^(1/T)``.
        """
        policy, value = self.run(game, tracker, add_noise=add_noise)
        size = game.board_size
        if policy.sum() == 0:
            return Move.pass_turn(), policy, value
        if temperature <= 1e-6:
            index = int(policy.argmax())
        else:
            weighted = np.power(policy, 1.0 / temperature)
            weighted /= weighted.sum()
            index = int(np.random.choice(len(weighted), p=weighted))
        return index_to_move(index, size), policy, value

    # -- internals ---------------------------------------------------------

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """PUCT: argmax over Q(child from parent's view) + U."""
        sqrt_n = math.sqrt(node.visit_count)
        best_score = -float("inf")
        best: Tuple[int, Node] = next(iter(node.children.items()))
        for index, child in node.children.items():
            # child.q is from the child mover's perspective; negate for parent.
            u = self.config.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = -child.q + u
            if score > best_score:
                best_score = score
                best = (index, child)
        return best

    def _expand(
        self, node: Node, game: Game, tracker: Optional[HistoryTracker]
    ) -> float:
        """Evaluate ``game`` with the network and create child edges."""
        planes = encode(
            game.board, tracker, game.current_player, self.config.history_planes
        )
        policy, value = self.net.predict(torch.from_numpy(planes))
        legal = game.legal_moves()
        size = game.board_size
        priors = np.zeros(self.config.policy_size, dtype=np.float64)
        for move in legal:
            index = move_to_index(move, size)
            priors[index] = float(policy[index])
        total = priors.sum()
        if total <= 1e-9:
            # Network assigns ~0 mass to all legal moves; use uniform priors.
            for move in legal:
                priors[move_to_index(move, size)] = 1.0
            total = priors.sum()
        priors /= total
        for index in np.nonzero(priors)[0]:
            node.children[int(index)] = Node(prior=float(priors[index]))
        return value

    def _add_dirichlet_noise(self, root: Node) -> None:
        """Mix Dirichlet noise into root priors (self-play exploration)."""
        indices = list(root.children.keys())
        if not indices:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(indices))
        eps = self.config.dirichlet_eps
        for index, n in zip(indices, noise):
            child = root.children[index]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    @staticmethod
    def _terminal_value(game: Game) -> float:
        """Value of a finished game for the side to move (usually -1)."""
        result = game.result
        assert result is not None
        if result.winner is game.current_player:
            return 1.0
        if result.winner is game.current_player.opponent:
            return -1.0
        return 0.0

    @staticmethod
    def _clone_tracker(tracker: Optional[HistoryTracker]) -> Optional[HistoryTracker]:
        if tracker is None:
            return None
        clone = HistoryTracker(tracker.history)
        for board in reversed(tracker.boards()):
            clone.push(board)
        return clone
