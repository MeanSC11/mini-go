"""Rank benchmark: calibrate a checkpoint against a reference engine.

Preferred path (accurate): play games against an engine of known rank (GNU Go,
Pachi, ... anything speaking GTP) and convert the win rate into a rank relative
to that reference.

Fallback (rough): if no reference engine is installed, report an ELO-based
estimate, clearly labelled as uncalibrated. The internal ELO comes from training
self-evaluation (pass it with ``--elo``); without it we can only state the
anchor assumption.

The reference engine is an *optional* dependency -- the project runs fine
without it; you just get the rougher estimate.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from typing import Callable, List, Optional, Tuple

from goengine import Color, Game, Move

from azero import rank as rank_mod
from azero.config import Config

logger = logging.getLogger(__name__)

_GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # GTP skips 'I'


def move_to_gtp(move: Move, size: int) -> str:
    if move.is_pass:
        return "pass"
    if move.is_resign:
        return "resign"
    assert move.point is not None
    row, col = move.point
    return f"{_GTP_COLS[col]}{size - row}"


def gtp_to_move(vertex: str, size: int) -> Move:
    v = vertex.strip().lower()
    if v == "pass":
        return Move.pass_turn()
    if v == "resign":
        return Move.resign()
    col = _GTP_COLS.lower().index(v[0])
    row = size - int(v[1:])
    return Move.play(row, col)


class GTPEngine:
    """Minimal GTP client over a subprocess (context manager)."""

    def __init__(self, command: List[str]) -> None:
        self.command = command
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "GTPEngine":
        self._proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        return self

    def _send(self, command: str) -> str:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                break
            if line.strip() == "" and lines:
                break
            lines.append(line.rstrip("\n"))
        text = "\n".join(lines).strip()
        if text.startswith("?"):
            raise RuntimeError(f"GTP error for {command!r}: {text}")
        return text[1:].strip() if text.startswith("=") else text

    def setup(self, size: int, komi: float) -> None:
        self._send(f"boardsize {size}")
        self._send("clear_board")
        self._send(f"komi {komi}")

    def genmove(self, color: str) -> str:
        return self._send(f"genmove {color}")

    def play(self, color: str, vertex: str) -> None:
        self._send(f"play {color} {vertex}")

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            try:
                self._send("quit")
            except Exception:
                pass
            self._proc.terminate()


def find_reference_engine(name: str = "gnugo") -> Optional[List[str]]:
    """Locate a known GTP engine; return its launch command or None."""
    if name == "gnugo" and shutil.which("gnugo"):
        return ["gnugo", "--mode", "gtp"]
    if name == "pachi" and shutil.which("pachi"):
        return ["pachi"]
    if shutil.which(name):
        return [name, "--mode", "gtp"]
    return None


MoveFn = Callable[[Game], Move]


def play_vs_reference(
    move_fn: MoveFn, engine_cmd: List[str], config: Config, games: int
) -> Tuple[int, int, int]:
    """Play ``games`` against the reference; return (wins, losses, draws).

    Our bot alternates colour each game. Scoring uses our engine (Chinese,
    matching training), so it is consistent across opponents.
    """
    size, komi = config.board_size, config.komi
    wins = losses = draws = 0
    for i in range(games):
        bot_color = Color.BLACK if i % 2 == 0 else Color.WHITE
        with GTPEngine(engine_cmd) as ref:
            ref.setup(size, komi)
            game = Game(size, komi)
            while not game.is_over and len(game.moves) < config.move_cap:
                if game.current_player is bot_color:
                    move = move_fn(game)
                    game.play(move)
                    if not move.is_resign:
                        ref.play("black" if bot_color is Color.BLACK else "white",
                                 move_to_gtp(move, size))
                    else:
                        break
                else:
                    ref_color = "white" if bot_color is Color.BLACK else "black"
                    vertex = ref.genmove(ref_color)
                    move = gtp_to_move(vertex, size)
                    if move.is_resign:
                        break  # reference resigned -> bot wins
                    game.play(move)
        winner = _winner(game, bot_color, resigned_move=move)
        if winner is Color.EMPTY:
            draws += 1
        elif winner is bot_color:
            wins += 1
        else:
            losses += 1
        logger.info("game %d/%d: %s", i + 1, games,
                    "win" if winner is bot_color else "loss/draw")
    return wins, losses, draws


def _winner(game: Game, bot_color: Color, resigned_move: Optional[Move]) -> Color:
    if resigned_move is not None and resigned_move.is_resign:
        # Whoever just resigned loses.
        return bot_color.opponent if game.current_player is bot_color else bot_color
    if game.result is not None:
        return game.result.winner
    black, white = game.score()
    if black > white:
        return Color.BLACK
    if white > black:
        return Color.WHITE
    return Color.EMPTY


def benchmark(
    checkpoint_path: str,
    device: str = "cuda",
    sims: int = 400,
    games: int = 20,
    reference: str = "gnugo",
    reference_rank: str = "6k",
    elo: Optional[float] = None,
) -> dict:
    """Estimate a checkpoint's rank; reference-calibrated if possible."""
    from azero.checkpoint import load_checkpoint, resolve_autocast, resolve_device
    from azero.features import HistoryTracker
    from azero.serve import CachingEvaluator, serve_search

    device = resolve_device(device)
    autocast_dtype, _ = resolve_autocast("auto", device)
    net, config, iteration = load_checkpoint(checkpoint_path, device)
    evaluator = CachingEvaluator(net, autocast_dtype)

    def move_fn(game: Game) -> Move:
        tracker = HistoryTracker(config.history_planes, Game(config.board_size, config.komi).board)
        replay = Game(config.board_size, config.komi)
        for m in game.moves:
            if m.is_resign:
                break
            replay.play(m)
            tracker.push(replay.board)
        move, _, _ = serve_search(
            evaluator, replay, tracker, config, max_simulations=sims, leaf_batch=8
        )
        return move

    engine_cmd = find_reference_engine(reference)
    if engine_cmd is not None:
        wins, losses, draws = play_vs_reference(move_fn, engine_cmd, config, games)
        win_rate = (wins + 0.5 * draws) / games if games else 0.0
        ref_g = rank_mod.parse_rank(reference_rank)
        g = rank_mod.reference_calibrated_g(ref_g, win_rate)
        return {
            "method": "reference-calibrated",
            "reference": reference,
            "reference_rank": reference_rank,
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "rank_label": rank_mod.g_to_label(g),
            "sims": sims,
            "iteration": iteration,
        }

    # Fallback: no reference engine installed.
    est_elo = elo if elo is not None else rank_mod.ANCHOR_ELO
    return {
        "method": "ELO estimate (uncalibrated)",
        "reference": None,
        "elo": est_elo,
        "rank_label": rank_mod.elo_to_label(est_elo),
        "sims": sims,
        "iteration": iteration,
        "note": (
            f"no '{reference}' engine found on PATH; install it for a "
            "reference-calibrated rank. This number assumes iteration-0 "
            f"~= {rank_mod.ANCHOR_KYU} kyu."
        ),
    }


def _format(result: dict) -> str:
    lines = [f"Rank benchmark ({result['method']}) — checkpoint iter {result['iteration']}"]
    if result["reference"]:
        lines.append(
            f"  vs {result['reference']} ({result['reference_rank']}): "
            f"{result['wins']}W/{result['losses']}L/{result['draws']}D "
            f"over {result['games']} games (win rate {result['win_rate']:.0%})"
        )
    else:
        lines.append(f"  internal ELO {result['elo']:.0f}")
        lines.append(f"  note: {result['note']}")
    lines.append(f"  estimated rank: {result['rank_label']}  (at {result['sims']} sims/move)")
    return "\n".join(lines)


def main() -> None:
    """CLI: benchmark a checkpoint's rank against a reference engine."""
    parser = argparse.ArgumentParser(description="Estimate a checkpoint's go rank")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=400, help="simulations/move for the bot")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--reference", default="gnugo", help="reference engine command name")
    parser.add_argument("--reference-rank", default="6k", help="assumed rank of the reference")
    parser.add_argument("--elo", type=float, default=None, help="internal ELO for the fallback")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = benchmark(
        args.checkpoint, device=args.device, sims=args.sims, games=args.games,
        reference=args.reference, reference_rank=args.reference_rank, elo=args.elo,
    )
    print(_format(result))


if __name__ == "__main__":
    main()
