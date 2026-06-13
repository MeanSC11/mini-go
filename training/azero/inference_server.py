"""Batched-inference self-play: CPU MCTS workers + one central GPU evaluator.

    worker 0 ─┐                         ┌─> response_q[0] ─> worker 0
    worker 1 ─┼─> request_q ─> server ──┼─> response_q[1] ─> worker 1
      ...     │   (model on GPU,        │      ...
    worker N ─┘    batches requests)    └─> response_q[N] ─> worker N

Each worker runs game-parallel MCTS (``azero.selfplay_engine``) and, every round,
sends all of its pending leaves as one request. The server accumulates requests
from every worker until it has ``inference_max_batch`` positions or
``inference_timeout_ms`` elapses, runs a single (optionally mixed-precision)
forward pass, and scatters the results back. All processes use the "spawn" start
method, so no worker ever inherits the parent's CUDA context.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
from typing import List, Optional, Tuple

import numpy as np

from azero.config import Config
from azero.replay import Sample
from azero.selfplay_engine import SelfPlayStats

logger = logging.getLogger(__name__)

_STOP = None  # sentinel pushed onto request_q to shut the server down


class RemoteEvaluator:
    """``evaluate(planes) -> (policies, values)`` backed by the GPU server."""

    def __init__(self, worker_id: int, request_q: "mp.Queue", response_q: "mp.Queue") -> None:
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q

    def __call__(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.ascontiguousarray(planes, dtype=np.float32)
        self.request_q.put((self.worker_id, arr))
        return self.response_q.get()


def _inference_server_loop(
    checkpoint_path: Optional[str],
    config_dict: dict,
    device: str,
    request_q: "mp.Queue",
    response_qs: List["mp.Queue"],
    stats_q: "mp.Queue",
) -> None:
    """Server process: hold the model, batch requests with a size/time cap."""
    import torch  # imported in the child so workers never import CUDA

    from azero.checkpoint import build_network, load_checkpoint, resolve_autocast

    config = Config(**config_dict)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    autocast_dtype, precision_label = resolve_autocast(config.precision, device)

    if checkpoint_path and Path(checkpoint_path).is_file():
        net, _, _ = load_checkpoint(checkpoint_path, device)
    else:
        net = build_network(config, device)
        net.eval()

    max_batch = max(1, config.inference_max_batch)
    timeout_s = max(0.0, config.inference_timeout_ms / 1000.0)
    logger.info(
        "inference server ready on %s (precision=%s, max_batch=%d, timeout=%.1fms)",
        device, precision_label, max_batch, config.inference_timeout_ms,
    )

    pending: List[Tuple[int, np.ndarray]] = []
    pend_total = 0
    deadline = 0.0
    total_calls = 0
    total_positions = 0
    max_seen = 0

    def fire() -> None:
        nonlocal pending, pend_total, total_calls, total_positions, max_seen
        if not pending:
            return
        arr = np.concatenate([a for _, a in pending], axis=0)
        policies, values = net.predict_many(arr, autocast_dtype)
        offset = 0
        for worker_id, a in pending:
            n = len(a)
            response_qs[worker_id].put((policies[offset:offset + n], values[offset:offset + n]))
            offset += n
        total_calls += 1
        total_positions += len(arr)
        max_seen = max(max_seen, len(arr))
        pending = []
        pend_total = 0

    while True:
        if not pending:
            item = request_q.get()
            if item is _STOP:
                break
            worker_id, arr = item
            pending.append((worker_id, arr))
            pend_total += len(arr)
            deadline = time.monotonic() + timeout_s
            continue

        if pend_total >= max_batch or time.monotonic() >= deadline:
            fire()
            continue

        try:
            item = request_q.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty:
            fire()
            continue
        if item is _STOP:
            fire()
            break
        worker_id, arr = item
        pending.append((worker_id, arr))
        pend_total += len(arr)

    fire()
    stats_q.put((total_calls, total_positions, max_seen))


def _selfplay_worker(
    worker_id: int,
    config_dict: dict,
    games: int,
    request_q: "mp.Queue",
    response_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """Worker process: game-parallel MCTS, offloading evaluation to the server."""
    try:
        # Deferred import keeps azero.selfplay <-> azero.inference_server acyclic.
        from azero.selfplay_engine import run_self_play

        config = Config(**config_dict)
        evaluator = RemoteEvaluator(worker_id, request_q, response_q)
        samples = run_self_play(
            evaluator, config, games, config.selfplay_concurrent_games
        )
        result_q.put((worker_id, samples))
    except BaseException as exc:  # report instead of hanging the parent
        import traceback

        traceback.print_exc()
        result_q.put((worker_id, exc))


def generate_games_server(
    checkpoint_path: Optional[str], config: Config, total_games: int, device: str
) -> Tuple[List[Sample], SelfPlayStats]:
    """Run ``total_games`` self-play games via CPU workers + a GPU server."""
    workers = max(1, min(config.workers, total_games))
    per_worker = [total_games // workers] * workers
    for i in range(total_games % workers):
        per_worker[i] += 1
    per_worker = [n for n in per_worker if n > 0]
    workers = len(per_worker)

    ctx = mp.get_context("spawn")
    request_q: "mp.Queue" = ctx.Queue()
    response_qs: List["mp.Queue"] = [ctx.Queue() for _ in range(workers)]
    result_q: "mp.Queue" = ctx.Queue()
    stats_q: "mp.Queue" = ctx.Queue()
    config_dict = config.to_dict()

    server = ctx.Process(
        target=_inference_server_loop,
        args=(checkpoint_path, config_dict, device, request_q, response_qs, stats_q),
        daemon=True,
    )
    server.start()

    procs: List["mp.Process"] = []
    for i, n in enumerate(per_worker):
        p = ctx.Process(
            target=_selfplay_worker,
            args=(i, config_dict, n, request_q, response_qs[i], result_q),
        )
        p.start()
        procs.append(p)

    samples: List[Sample] = []
    collected = 0
    try:
        while collected < workers:
            try:
                worker_id, payload = result_q.get(timeout=60)
            except Empty:
                if server.exitcode is not None:
                    raise RuntimeError(
                        f"inference server exited early (code {server.exitcode})"
                    )
                dead = [p for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(
                        f"self-play worker crashed (exit code {dead[0].exitcode})"
                    )
                continue
            collected += 1
            if isinstance(payload, BaseException):
                raise RuntimeError(f"self-play worker {worker_id} failed: {payload!r}")
            samples.extend(payload)
    finally:
        request_q.put(_STOP)
        for p in procs:
            p.join(timeout=30)
        server.join(timeout=30)
        for p in [*procs, server]:
            if p.is_alive():
                p.terminate()

    try:
        calls, positions, _max_seen = stats_q.get(timeout=5)
    except Empty:
        calls, positions = 0, 0
    stats = SelfPlayStats(games=total_games, positions=positions, forward_calls=calls)
    return samples, stats
