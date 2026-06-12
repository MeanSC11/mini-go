"""Head-to-head evaluation between two networks."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Tuple

from goengine import Color, Game

from azero.checkpoint import load_checkpoint, resolve_device
from azero.config import Config
from azero.features import HistoryTracker
from azero.mcts import MCTS
from azero.network import PolicyValueNet

logger = logging.getLogger(__name__)


def play_match_game(
    black_net: PolicyValueNet, white_net: PolicyValueNet, config: Config
) -> Color:
    """Play one evaluation game (greedy moves, no noise); return the winner."""
    eval_config = dataclasses.replace(config, simulations=config.eval_simulations)
    game = Game(config.board_size, config.komi)
    tracker = HistoryTracker(config.history_planes, game.board)
    players = {Color.BLACK: MCTS(black_net, eval_config),
               Color.WHITE: MCTS(white_net, eval_config)}
    for _ in range(config.move_cap):
        if game.is_over:
            break
        mcts = players[game.current_player]
        move, _, _ = mcts.choose_move(game, tracker, temperature=0.0, add_noise=False)
        game.play(move)
        tracker.push(game.board)
    if game.result is not None:
        return game.result.winner
    black, white = game.score()
    return Color.BLACK if black > white else Color.WHITE


def evaluate(
    challenger: PolicyValueNet,
    incumbent: PolicyValueNet,
    config: Config,
    games: int,
) -> Tuple[float, int, int]:
    """Pit challenger vs incumbent, alternating colors.

    Returns ``(challenger_win_rate, wins, losses)``; draws count half.
    """
    wins = 0
    losses = 0
    draws = 0
    for i in range(games):
        if i % 2 == 0:
            winner = play_match_game(challenger, incumbent, config)
            challenger_color = Color.BLACK
        else:
            winner = play_match_game(incumbent, challenger, config)
            challenger_color = Color.WHITE
        if winner is Color.EMPTY:
            draws += 1
        elif winner is challenger_color:
            wins += 1
        else:
            losses += 1
    win_rate = (wins + 0.5 * draws) / games if games else 0.0
    return win_rate, wins, losses


def main() -> None:
    """CLI: evaluate two checkpoints against each other."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoint A vs B")
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = Config.load(args.config)
    device = resolve_device(config.device)
    net_a, _, _ = load_checkpoint(args.checkpoint_a, device)
    net_b, _, _ = load_checkpoint(args.checkpoint_b, device)
    win_rate, wins, losses = evaluate(net_a, net_b, config, args.games)
    logger.info(
        "A vs B: win rate %.1f%% (%d wins, %d losses, %d games)",
        100 * win_rate, wins, losses, args.games,
    )


if __name__ == "__main__":
    main()
