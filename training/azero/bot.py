"""Play interface used by the inference service to serve trained models."""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

from goengine import Game, Move

from azero.checkpoint import load_checkpoint, resolve_device
from azero.features import HistoryTracker
from azero.mcts import MCTS


class AlphaZeroPlayer:
    """A bot backed by one checkpoint; thread-safe for sequential calls."""

    def __init__(
        self, checkpoint_path: str, simulations: int = 160, device: str = "cpu"
    ) -> None:
        device = resolve_device(device)
        net, config, iteration = load_checkpoint(checkpoint_path, device)
        self.iteration = iteration
        self.config = dataclasses.replace(
            config, simulations=simulations, device=device
        )
        self.net = net
        self._mcts = MCTS(self.net, self.config)

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
        move, policy, value = self._mcts.choose_move(
            replay, tracker, temperature=0.0, add_noise=False
        )
        size = game.board_size
        policy_dict: Dict[str, float] = {}
        for index, prob in enumerate(policy):
            if prob <= 0:
                continue
            if index == size * size:
                policy_dict["pass"] = float(prob)
            else:
                policy_dict[f"{index // size},{index % size}"] = float(prob)
        win_rate = (1.0 + value) / 2.0  # tanh value -> probability
        return move, win_rate, policy_dict
