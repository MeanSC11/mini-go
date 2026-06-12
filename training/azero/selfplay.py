"""Self-play game generation, single-process and multi-process."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

from goengine import Color, Game

from azero.checkpoint import build_network, load_checkpoint, resolve_device
from azero.config import Config
from azero.features import HistoryTracker, encode
from azero.mcts import MCTS
from azero.network import PolicyValueNet
from azero.replay import Sample

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


def _worker_play(args: tuple) -> List[Sample]:
    """Process-pool entry point: load weights from disk and play games."""
    checkpoint_path, config_dict, games = args
    config = Config(**config_dict)
    device = resolve_device(config.device)
    if checkpoint_path and Path(checkpoint_path).is_file():
        net, _, _ = load_checkpoint(checkpoint_path, device)
    else:
        net = build_network(config, device)
        net.eval()
    samples: List[Sample] = []
    for _ in range(games):
        samples.extend(play_game(net, config))
    return samples


def generate_games(
    checkpoint_path: Optional[str], config: Config, total_games: int
) -> List[Sample]:
    """Generate ``total_games`` self-play games across worker processes."""
    workers = max(1, config.workers)
    if workers == 1:
        return _worker_play((checkpoint_path, config.to_dict(), total_games))
    per_worker = [total_games // workers] * workers
    for i in range(total_games % workers):
        per_worker[i] += 1
    tasks = [
        (checkpoint_path, config.to_dict(), n) for n in per_worker if n > 0
    ]
    samples: List[Sample] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_worker_play, tasks):
            samples.extend(result)
    return samples


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
    samples = generate_games(args.checkpoint, config, games)
    states = np.stack([s[0] for s in samples])
    policies = np.stack([s[1] for s in samples])
    values = np.asarray([s[2] for s in samples], dtype=np.float32)
    np.savez_compressed(args.out, states=states, policies=policies, values=values)
    logger.info("wrote %d samples from %d games to %s", len(samples), games, args.out)


if __name__ == "__main__":
    main()
