"""Self-play game generation, single-process and multi-process."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from goengine import Color, Game

from azero.batched_selfplay import generate_games_batched
from azero.checkpoint import (
    build_network,
    load_checkpoint,
    resolve_autocast,
    resolve_device,
)
from azero.inference_server import generate_games_server
from azero.config import Config
from azero.features import HistoryTracker, encode
from azero.mcts import MCTS
from azero.network import PolicyValueNet
from azero.replay import Sample
from azero.selfplay_engine import SelfPlayStats

logger = logging.getLogger(__name__)


def play_game(net: PolicyValueNet, config: Config) -> List[Sample]:
    """Play one self-play game; return (state, policy, z) training samples."""
    game = Game(config.board_size, config.komi)
    tracker = HistoryTracker(config.history_planes, game.board)
    mcts = MCTS(net, config)
    pending: List[tuple] = []  # (planes, policy, player)

    for move_number in range(config.move_cap):
        if game.is_over:
            break
        planes = encode(
            game.board, tracker, game.current_player, config.history_planes
        )
        temperature = 1.0 if move_number < config.temperature_moves else 0.0
        move, policy, _ = mcts.choose_move(
            game, tracker, temperature=temperature, add_noise=True
        )
        pending.append((planes, policy, game.current_player))
        game.play(move)
        tracker.push(game.board)

    winner = _winner(game)
    samples: List[Sample] = []
    for planes, policy, player in pending:
        if winner is Color.EMPTY:
            z = 0.0
        else:
            z = 1.0 if winner is player else -1.0
        samples.append((planes, policy, z))
    return samples


def _winner(game: Game) -> Color:
    """Winner of a (possibly move-capped) game by area score."""
    if game.result is not None:
        return game.result.winner
    black, white = game.score()
    if black > white:
        return Color.BLACK
    if white > black:
        return Color.WHITE
    return Color.EMPTY


def _load_selfplay_net(
    checkpoint_path: Optional[str], config: Config, device: str
) -> PolicyValueNet:
    """Load the self-play network from a checkpoint, or a fresh one."""
    if checkpoint_path and Path(checkpoint_path).is_file():
        net, _, _ = load_checkpoint(checkpoint_path, device)
    else:
        net = build_network(config, device)
        net.eval()
    return net


def generate_games(
    checkpoint_path: Optional[str], config: Config, total_games: int
) -> Tuple[List[Sample], SelfPlayStats]:
    """Generate ``total_games`` self-play games; return ``(samples, stats)``.

    With ``workers >= 2`` self-play runs as game-parallel MCTS worker processes
    feeding a single inference server that batches their leaf evaluations across
    workers (see :mod:`azero.inference_server`) -- tree search spreads across
    cores while the GPU sees large batches. ``workers == 1`` runs a single
    in-process game-parallel generator (good for one-GPU boxes).
    """
    device = resolve_device(config.selfplay_device)
    workers = max(1, min(config.workers, total_games))

    if workers >= 2:
        return generate_games_server(checkpoint_path, config, total_games, device)

    net = _load_selfplay_net(checkpoint_path, config, device)
    autocast_dtype, _ = resolve_autocast(config.precision, device)
    stats = SelfPlayStats()
    samples = generate_games_batched(net, config, total_games, autocast_dtype, stats)
    return samples, stats


def main() -> None:
    """CLI: generate self-play games and write them to an .npz file."""
    parser = argparse.ArgumentParser(description="Generate self-play data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint (.pt)")
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--out", default="selfplay.npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = Config.load(args.config)
    games = args.games or config.games_per_iteration
    samples, stats = generate_games(args.checkpoint, config, games)
    logger.info("self-play avg batch %.1f over %d forwards", stats.avg_batch, stats.forward_calls)
    states = np.stack([s[0] for s in samples])
    policies = np.stack([s[1] for s in samples])
    values = np.asarray([s[2] for s in samples], dtype=np.float32)
    np.savez_compressed(args.out, states=states, policies=policies, values=values)
    logger.info("wrote %d samples from %d games to %s", len(samples), games, args.out)


if __name__ == "__main__":
    main()
