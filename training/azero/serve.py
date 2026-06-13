"""Serve-time play: fast, strong, and self-aware about the hardware it runs on.

This module is entirely separate from the training loop. It provides:

* :class:`CachingEvaluator` -- a transposition/eval cache in front of a network's
  batched (optionally BF16/FP16) forward pass, so repeated positions are free.
* :func:`serve_search` -- a single-move MCTS that runs as many simulations as a
  ``time_limit`` and/or ``max_simulations`` budget allows. Strength at serve time
  is governed purely by how many simulations we afford -- the network is
  unchanged, it just thinks deeper. This is fully decoupled from the (small,
  speed-tuned) self-play simulation count used during training.
* :func:`detect_hardware`, :func:`measure_sims_per_sec`, :func:`assess_resources`
  -- detect the GPU/CPU, micro-benchmark search throughput, and estimate the
  rank this machine can reach at a given time/move.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from goengine import Game, Move

from azero import rank as rank_mod
from azero.config import Config
from azero.features import HistoryTracker, index_to_move
from azero.selfplay_engine import _GameSearch

logger = logging.getLogger(__name__)

EvaluateFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


class CachingEvaluator:
    """Batched network evaluation with an exact transposition (eval) cache.

    The cache key is the encoded position's exact bytes (history included), so a
    hit always returns the correct evaluation -- no approximation. Bounded so it
    never grows without limit on long games / weak hardware.
    """

    def __init__(self, net, autocast_dtype=None, max_entries: int = 300_000) -> None:
        self.net = net
        self.autocast_dtype = autocast_dtype
        self.max_entries = max_entries
        self._cache: Dict[bytes, Tuple[np.ndarray, float]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(planes)
        out_p: list = [None] * n
        out_v = np.zeros(n, dtype=np.float32)
        todo_idx = []
        todo_planes = []
        for i in range(n):
            key = planes[i].tobytes()
            cached = self._cache.get(key)
            if cached is None:
                todo_idx.append(i)
                todo_planes.append(planes[i])
            else:
                out_p[i], out_v[i] = cached
                self.hits += 1
        if todo_planes:
            pol, val = self.net.predict_many(np.stack(todo_planes), self.autocast_dtype)
            self.misses += len(todo_planes)
            for j, i in enumerate(todo_idx):
                out_p[i] = pol[j]
                out_v[i] = float(val[j])
                if len(self._cache) < self.max_entries:
                    self._cache[planes[i].tobytes()] = (pol[j], float(val[j]))
        return np.stack(out_p), out_v


def serve_search(
    evaluate: EvaluateFn,
    game: Game,
    tracker: HistoryTracker,
    config: Config,
    *,
    max_simulations: Optional[int] = None,
    time_limit: Optional[float] = None,
    leaf_batch: int = 8,
) -> Tuple[Move, float, np.ndarray]:
    """Search one move; return ``(move, win_rate_side_to_move, visit_policy)``.

    Runs until ``max_simulations`` descents or ``time_limit`` seconds, whichever
    comes first (at least one must be given). Greedy, no Dirichlet noise.
    """
    if max_simulations is None and time_limit is None:
        max_simulations = config.simulations
    cap = max_simulations if max_simulations is not None else 1_000_000_000

    gs = _GameSearch(
        config, simulations=cap, add_noise=False, record_samples=False, greedy=True,
    )
    # Reuse the live game's position/history rather than _GameSearch's fresh board.
    gs.game = game.copy()
    gs.tracker = _copy_tracker(tracker)
    gs._start_move()

    deadline = (time.monotonic() + time_limit) if time_limit else None
    saved_leaf_batch = config.mcts_leaf_batch
    config.mcts_leaf_batch = max(1, leaf_batch)
    try:
        while True:
            planes = gs.collect()
            if planes:
                policies, values = evaluate(np.stack(planes))
                gs.consume(policies, values)
            if not gs.root_expanded:
                continue
            if gs.move_ready():
                break
            if max_simulations is not None and gs.descents >= max_simulations:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
    finally:
        config.mcts_leaf_batch = saved_leaf_batch

    size = config.board_size
    visits = np.zeros(config.policy_size, dtype=np.float32)
    for index, child in gs.root.children.items():
        visits[index] = child.visit_count
    total = visits.sum()
    if total > 0:
        visits /= total
    move = Move.pass_turn() if total == 0 else index_to_move(int(visits.argmax()), size)
    win_rate = (1.0 + gs.root.q) / 2.0  # root q is the side-to-move expected value
    return move, win_rate, visits


def _copy_tracker(tracker: HistoryTracker) -> HistoryTracker:
    clone = HistoryTracker(tracker.history)
    for board in reversed(tracker.boards()):
        clone.push(board)
    return clone


# --- hardware / throughput / rank assessment ---------------------------------


def detect_hardware() -> dict:
    """Best-effort hardware probe (never raises)."""
    import torch

    info = {
        "cpu_cores": os.cpu_count() or 1,
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": 0.0,
        "bf16": False,
    }
    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                device="cuda",
                gpu_name=props.name,
                vram_gb=round(props.total_memory / 1e9, 1),
                bf16=torch.cuda.is_bf16_supported(),
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("hardware probe failed: %r", exc)
    return info


def measure_sims_per_sec(
    net, config: Config, seconds: float = 12.0, leaf_batch: int = 8, autocast_dtype=None
) -> float:
    """Micro-benchmark: MCTS simulations/second from the opening position.

    Cache is intentionally OFF here so we measure raw search throughput (a
    realistic lower bound), not inflated by transpositions on the empty board.
    """
    game = Game(config.board_size, config.komi)
    tracker = HistoryTracker(config.history_planes, game.board)

    def evaluate(planes: np.ndarray):
        return net.predict_many(planes, autocast_dtype)

    cap = max(1, config.board_size * config.board_size)  # high enough to fill the time
    gs = _GameSearch(config, simulations=10_000_000, add_noise=False,
                     record_samples=False, greedy=True)
    gs.game = game
    gs.tracker = tracker
    gs._start_move()
    saved = config.mcts_leaf_batch
    config.mcts_leaf_batch = max(1, leaf_batch)
    start = time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            planes = gs.collect()
            if planes:
                policies, values = evaluate(np.stack(planes))
                gs.consume(policies, values)
            if gs.move_ready():  # finished a (huge) move budget; start another
                gs._start_move()
            _ = cap  # keep flake-free
    finally:
        config.mcts_leaf_batch = saved
    elapsed = time.monotonic() - start
    return gs.descents / elapsed if elapsed > 0 else 0.0


def assess_resources(
    checkpoint_path: str,
    device: str = "cuda",
    time_per_move: float = 5.0,
    bench_seconds: float = 12.0,
    leaf_batch: int = 8,
    base_rank: str = "25k",
    calibrated_rank: Optional[float] = None,
    calibrated_sims: Optional[float] = None,
) -> dict:
    """Probe hardware, benchmark throughput, and estimate the playable rank.

    ``calibrated_rank``/``calibrated_sims`` (from the reference benchmark) anchor
    the rank precisely; otherwise we fall back to an uncalibrated heuristic
    anchored at ``base_rank``.
    """
    from azero.checkpoint import load_checkpoint, resolve_autocast, resolve_device

    device = resolve_device(device)
    autocast_dtype, precision = resolve_autocast("auto", device)
    net, config, iteration = load_checkpoint(checkpoint_path, device)
    hw = detect_hardware()

    sims_per_sec = measure_sims_per_sec(
        net, config, seconds=bench_seconds, leaf_batch=leaf_batch,
        autocast_dtype=autocast_dtype,
    )
    sims_at_time = sims_per_sec * time_per_move

    if calibrated_rank is not None and calibrated_sims:
        base_g, base_sims, method = calibrated_rank, calibrated_sims, "reference-calibrated"
    else:
        base_g, base_sims, method = rank_mod.parse_rank(base_rank), 800.0, "uncalibrated heuristic"
    g_at_time = rank_mod.sims_to_g(sims_at_time, base_g, base_sims=base_sims)

    result = {
        "hardware": hw,
        "precision": precision,
        "board_size": config.board_size,
        "checkpoint_iteration": iteration,
        "sims_per_sec": sims_per_sec,
        "time_per_move": time_per_move,
        "sims_at_time": sims_at_time,
        "rank_label": rank_mod.g_to_label(g_at_time),
        "rank_method": method,
    }
    return result


def _format_assessment(a: dict) -> str:
    hw = a["hardware"]
    if hw["device"] == "cuda":
        hw_str = f"GPU {hw['gpu_name']} ({hw['vram_gb']} GB), {hw['cpu_cores']} CPU cores"
    else:
        hw_str = f"CPU only, {hw['cpu_cores']} cores"
    return (
        f"Detected: {hw_str}\n"
        f"Precision: {a['precision']}   Board: {a['board_size']}x{a['board_size']}   "
        f"Checkpoint: iter {a['checkpoint_iteration']}\n"
        f"This machine runs ~{a['sims_per_sec']:.0f} MCTS sims/second\n"
        f"At {a['time_per_move']:.1f}s/move -> ~{a['sims_at_time']:.0f} sims/move\n"
        f"Estimated playable strength: {a['rank_label']}  "
        f"({a['rank_method']})\n"
        f"Tune --time per move to trade speed for strength (more time = stronger)."
    )


def main() -> None:
    """CLI: assess this machine's playable rank for a checkpoint."""
    parser = argparse.ArgumentParser(description="Assess serve hardware -> rank")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time", type=float, default=5.0, help="seconds per move")
    parser.add_argument("--bench-seconds", type=float, default=12.0)
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--base-rank", default="25k",
                        help="uncalibrated anchor rank at 800 sims (e.g. '8k', '2d')")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = assess_resources(
        args.checkpoint, device=args.device, time_per_move=args.time,
        bench_seconds=args.bench_seconds, leaf_batch=args.leaf_batch,
        base_rank=args.base_rank,
    )
    print(_format_assessment(result))


if __name__ == "__main__":
    main()
