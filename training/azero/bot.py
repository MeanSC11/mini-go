"""Play interface used by the inference service to serve trained models.

Serving is decoupled from training: strength here is governed by ``simulations``
(or a per-move ``time_limit``), not the small self-play count. Evaluations run
through a :class:`~azero.serve.CachingEvaluator` (transposition cache + optional
BF16/FP16) and a batched, leaf-parallel search.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

from goengine import Game, Move

from azero.checkpoint import load_checkpoint, resolve_autocast, resolve_device
from azero.features import HistoryTracker
from azero.serve import CachingEvaluator, serve_search


class AlphaZeroPlayer:
    """A bot backed by one checkpoint; thread-safe for sequential calls."""

    def __init__(
        self,
        checkpoint_path: str,
        simulations: int = 160,
        device: str = "cpu",
        time_limit: Optional[float] = None,
        leaf_batch: int = 8,
        precision: str = "auto",
    ) -> None:
        device = resolve_device(device)
        net, config, iteration = load_checkpoint(checkpoint_path, device)
        self.iteration = iteration
        self.simulations = simulations
        self.time_limit = time_limit  # seconds/move; overrides simulations when set
        self.leaf_batch = leaf_batch
        self.config = dataclasses.replace(config, device=device)
        autocast_dtype, _ = resolve_autocast(precision, device)
        self.net = net
        self._evaluator = CachingEvaluator(net, autocast_dtype)

    def search(self, game: Game) -> Tuple[Move, Optional[float], Dict[str, float]]:
        """Choose a move; returns (move, side-to-move win rate, policy dict).

        The board size of ``game`` must match the checkpoint's board size.
        """
        if game.board_size != self.config.board_size:
            raise ValueError(
                f"checkpoint is for {self.config.board_size}x"
                f"{self.config.board_size}, got {game.board_size}"
            )
        tracker = HistoryTracker(self.config.history_planes, Game(
            game.board_size, game.komi
        ).board)
        # Rebuild history snapshots by replaying the move list.
        replay = Game(game.board_size, game.komi)
        for move in game.moves:
            if move.is_resign:
                break
            replay.play(move)
            tracker.push(replay.board)

        move, win_rate, visits = serve_search(
            self._evaluator, replay, tracker, self.config,
            max_simulations=None if self.time_limit else self.simulations,
            time_limit=self.time_limit,
            leaf_batch=self.leaf_batch,
        )

        size = game.board_size
        policy_dict: Dict[str, float] = {}
        for index, prob in enumerate(visits):
            if prob <= 0:
                continue
            if index == size * size:
                policy_dict["pass"] = float(prob)
            else:
                policy_dict[f"{index // size},{index % size}"] = float(prob)
        return move, win_rate, policy_dict
