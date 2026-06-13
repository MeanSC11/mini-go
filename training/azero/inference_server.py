"""Batched-inference self-play: CPU MCTS workers + one central GPU evaluator.

Architecture (all processes use the "spawn" start method, so no worker ever
inherits the parent's CUDA context):

    worker 0 ─┐                         ┌─> response_q[0] ─> worker 0
    worker 1 ─┼─> request_q ─> server ──┼─> response_q[1] ─> worker 1
      ...     │   (holds the model      │      ...
    worker N ─┘    on the GPU)          └─> response_q[N] ─> worker N

Each worker runs MCTS for its share of games (pure-Python, CPU-bound) and, for
every leaf it needs evaluated, sends the encoded planes to the server and blocks
for the result. The server drains all requests currently queued, stacks them
into one batch, runs a single GPU forward pass, and scatters the results back.

This fixes both problems of the single-process design:
  * MCTS now runs across N cores instead of one (no more one core at 100%);
  * the GPU sees real batches instead of batch-1 calls (no more ~1% util).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import List, Optional

import numpy as np

from azero.config import Config
from azero.replay import Sample

logger = logging.getLogger(__name__)

# Sentinel pushed onto the request queue to tell the server to shut down.
_STOP = None


class RemoteEvaluator:
    """Drop-in for ``PolicyValueNet.predict`` backed by the inference server.

    :class:`azero.mcts.MCTS` only ever calls ``net.predict(tensor)``, so a worker
    can hand it one of these instead of a real network and every leaf evaluation
    is transparently offloaded to the GPU server.
    """

    def __init__(self, worker_id: int, request_q: "mp.Queue", response_q: "mp.Queue") -> None:
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q

    def predict(self, planes):
        # ``planes`` is the torch tensor MCTS built via torch.from_numpy(...).
        if hasattr(planes, "detach"):
            arr = planes.detach().cpu().numpy()
        else:
            arr = np.asarray(planes, dtype=np.float32)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.request_q.put((self.worker_id, arr))
        policy, value = self.response_q.get()
        return policy, value


def _inference_server_loop(
    checkpoint_path: Optional[str],
    config_dict: dict,
    device: str,
    request_q: "mp.Queue",
    response_qs: List["mp.Queue"],
    max_batch: int,
) -> None:
    """Server process: hold the model on ``device``, batch and answer requests."""
    import torch  # imported in the child so workers never import CUDA

    from azero.checkpoint import build_network, load_checkpoint

    config = Config(**config_dict)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if checkpoint_path and Path(checkpoint_path).is_file():
        net, _, _ = load_checkpoint(checkpoint_path, device)
    else:
        net = build_network(config, device)
        net.eval()
    logger.info("inference server ready on %s (max_batch=%d)", device, max_batch)

    while True:
        first = request_q.get()
        if first is _STOP:
            break
        batch = [first]
        stop = False
        while len(batch) < max_batch:
            try:
                item = request_q.get_nowait()
            except Empty:
                break
            if item is _STOP:
                stop = True
                break
            batch.append(item)

        planes = np.stack([arr for _, arr in batch])
        policies, values = net.predict_many(planes)
        for (worker_id, _), policy, value in zip(batch, policies, values):
            response_qs[worker_id].put((policy, float(value)))

        if stop:
            break


def _selfplay_worker(
    worker_id: int,
    config_dict: dict,
    games: int,
    request_q: "mp.Queue",
    response_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """Worker process: run MCTS self-play, offloading evaluation to the server."""
    try:
        # Deferred import keeps azero.selfplay <-> azero.inference_server acyclic.
        from azero.selfplay import play_game

        config = Config(**config_dict)
        evaluator = RemoteEvaluator(worker_id, request_q, response_q)
        samples: List[Sample] = []
        for _ in range(games):
            samples.extend(play_game(evaluator, config))
        result_q.put((worker_id, samples))
    except BaseException as exc:  # report instead of hanging the parent
        import traceback

        traceback.print_exc()
        result_q.put((worker_id, exc))


def generate_games_server(
    checkpoint_path: Optional[str], config: Config, total_games: int, device: str
) -> List[Sample]:
    """Run ``total_games`` self-play games via CPU workers + a GPU server."""
    workers = max(1, min(config.workers, total_games))
    per_worker = [total_games // workers] * workers
    for i in range(total_games % workers):
        per_worker[i] += 1
    per_worker = [n for n in per_worker if n > 0]
    workers = len(per_worker)
    max_batch = config.selfplay_parallel_games or max(workers, 64)

    ctx = mp.get_context("spawn")
    request_q: "mp.Queue" = ctx.Queue()
    response_qs: List["mp.Queue"] = [ctx.Queue() for _ in range(workers)]
    result_q: "mp.Queue" = ctx.Queue()
    config_dict = config.to_dict()

    server = ctx.Process(
        target=_inference_server_loop,
        args=(checkpoint_path, config_dict, device, request_q, response_qs, max_batch),
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
                worker_id, payload = result_q.get(timeout=30)
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
        request_q.put(_STOP)  # ask the server to drain and exit
        for p in procs:
            p.join(timeout=30)
        server.join(timeout=30)
        for p in [*procs, server]:
            if p.is_alive():
                p.terminate()

    return samples
