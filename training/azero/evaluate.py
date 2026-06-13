"""Head-to-head evaluation between two networks."""

from __future__ import annotations

import argparse
import logging
from typing import Tuple

from goengine import Color

from azero.checkpoint import load_checkpoint, resolve_autocast, resolve_device
from azero.config import Config
from azero.network import PolicyValueNet
from azero.selfplay_engine import run_evaluation

logger = logging.getLogger(__name__)


def evaluate(
    challenger: PolicyValueNet,
    incumbent: PolicyValueNet,
    config: Config,
    games: int,
) -> Tuple[float, int, int]:
    """Pit challenger vs incumbent, alternating colors, with batched inference.

    All ``games`` run concurrently; each round buckets pending leaves by which
    network the side to move uses and fires one batched forward per network --
    the same speed-up self-play gets, applied to evaluation. Moves are greedy
    with no Dirichlet noise (deterministic match play). Returns
    ``(challenger_win_rate, wins, losses)``; draws count half.
    """
    device = next(challenger.parameters()).device
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype, _ = resolve_autocast(config.precision, dev_str)
    # key 0 -> challenger, key 1 -> incumbent.
    evaluate_fns = (
        lambda arr: challenger.predict_many(arr, autocast_dtype),
        lambda arr: incumbent.predict_many(arr, autocast_dtype),
    )
    specs = [
        {Color.BLACK: 0, Color.WHITE: 1} if i % 2 == 0
        else {Color.BLACK: 1, Color.WHITE: 0}
        for i in range(games)
    ]
    concurrent = min(games, max(1, config.selfplay_concurrent_games))
    winners = run_evaluation(
        evaluate_fns, config, specs, concurrent, config.eval_simulations
    )

    wins = losses = draws = 0
    for i, winner in enumerate(winners):
        challenger_color = Color.BLACK if i % 2 == 0 else Color.WHITE
        if winner is None or winner is Color.EMPTY:
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
