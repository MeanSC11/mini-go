"""AlphaZero training loop.

Each iteration:
  1. generate self-play games with the current best model (parallel workers)
  2. add samples to the replay buffer
  3. train the candidate network for N steps
  4. evaluate candidate vs best; promote if it wins enough
  5. write a checkpoint and TensorBoard logs
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from azero.checkpoint import (
    build_network,
    load_checkpoint,
    resolve_autocast,
    resolve_device,
    save_checkpoint,
)
from azero.config import Config
from azero.evaluate import evaluate
from azero.plateau import PlateauController, TrainingState, final_report
from azero.replay import ReplayBuffer
from azero.selfplay import generate_games

logger = logging.getLogger(__name__)


class _GPUSampler:
    """Average GPU utilization over a block by polling NVML in a background thread."""

    def __init__(self, device: str, interval: float = 0.2) -> None:
        self.device = device
        self.interval = interval
        self._samples: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None

    def __enter__(self) -> "_GPUSampler":
        if self.device.startswith("cuda"):
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
            except Exception as exc:  # missing pynvml/driver -> report n/a
                logger.debug("GPU utilization sampling unavailable: %r", exc)
                self._handle = None
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._samples.append(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            except Exception:
                return

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    @property
    def average(self) -> Optional[float]:
        return sum(self._samples) / len(self._samples) if self._samples else None


def train_steps(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    config: Config,
    device: str,
    autocast_dtype=None,
    scaler: "Optional[torch.amp.GradScaler]" = None,
) -> tuple[float, float]:
    """Run the configured number of SGD steps; return mean (policy, value) loss."""
    net.train()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_autocast = autocast_dtype is not None and device_type == "cuda"
    policy_losses = []
    value_losses = []
    for _ in range(config.train_steps_per_iteration):
        states, policies, values = buffer.sample(config.batch_size)
        x = torch.from_numpy(states).to(device)
        target_policy = torch.from_numpy(policies).to(device)
        target_value = torch.from_numpy(values).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
            logits, predicted_value = net(x)
            # Losses in fp32 for numerical stability under mixed precision.
            log_probs = F.log_softmax(logits.float(), dim=1)
            policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_value.float(), target_value)
            loss = policy_loss + config.value_loss_weight * value_loss

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def run(config: Config, resume: bool = False) -> None:
    """Execute the full training loop."""
    device = resolve_device(config.device)
    autocast_dtype, precision_label = resolve_autocast(config.precision, device)
    if device.startswith("cuda"):
        # Let cuDNN pick the fastest convs for the fixed input shape and use
        # TF32 matmuls — free throughput on Ampere/Hopper (e.g. H100).
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    scaler = torch.amp.GradScaler("cuda", enabled=(precision_label == "fp16"))
    logger.info("training on device: %s (precision=%s)", device, precision_label)
    run_dir = Path(config.run_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    buffer_path = run_dir / "replay.npz"
    writer = SummaryWriter(log_dir=str(run_dir))

    start_iteration = 0
    if resume and best_path.is_file():
        candidate, _, start_iteration = load_checkpoint(best_path, device)
        logger.info("resumed from %s at iteration %d", best_path, start_iteration)
    else:
        candidate = build_network(config, device)
        save_checkpoint(candidate, config, 0, best_path)

    buffer = ReplayBuffer(config.buffer_size)
    if resume and buffer_path.is_file():
        buffer.load(buffer_path)
        logger.info("restored replay buffer with %d samples", len(buffer))

    optimizer = torch.optim.Adam(
        candidate.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    state_path = run_dir / "training_state.json"
    state = TrainingState.load(state_path) if resume else TrainingState()
    controller = PlateauController(config, state)
    if resume:
        # Continue past whatever finished last; never redo iterations.
        start_iteration = max(start_iteration, state.iteration)
        logger.info("resumed plateau state: best ELO %.0f (%s), mode=%s, iter=%d",
                    state.best_elo, controller.rank_label, state.mode, state.iteration)

    for iteration in range(start_iteration + 1, config.iterations + 1):
        if controller.elapsed_hours() >= config.max_hours:
            state.stop_reason = "max_hours"
            logger.info("max_hours (%.1f h) reached before iteration %d",
                        config.max_hours, iteration)
            break

        started = time.time()
        # Self-play uses shake-boosted exploration while escaping a plateau.
        selfplay_config = controller.effective_selfplay_config()
        for group in optimizer.param_groups:
            group["lr"] = controller.current_lr()
        with _GPUSampler(device) as gpu:
            # 1-2. Self-play with the current best model.
            sp_start = time.time()
            samples, sp_stats = generate_games(
                str(best_path), selfplay_config, config.games_per_iteration
            )
            sp_time = max(time.time() - sp_start, 1e-6)
            games_per_sec = config.games_per_iteration / sp_time
            buffer.add(samples)
            buffer.save(buffer_path)
            writer.add_scalar("selfplay/samples", len(samples), iteration)
            writer.add_scalar("selfplay/buffer_size", len(buffer), iteration)
            writer.add_scalar("selfplay/games_per_sec", games_per_sec, iteration)
            writer.add_scalar("selfplay/avg_batch", sp_stats.avg_batch, iteration)
            writer.add_scalar("time/selfplay_sec", sp_time, iteration)

            if len(buffer) < config.min_buffer_size:
                logger.info(
                    "iteration %d: buffer %d < %d, skipping training "
                    "(self-play %.0fs, %.1f games/s, avg batch %.0f)",
                    iteration, len(buffer), config.min_buffer_size,
                    sp_time, games_per_sec, sp_stats.avg_batch,
                )
                state.iteration = iteration
                controller.tick_time()
                state.save(state_path)
                continue

            # 3. Train the candidate.
            train_start = time.time()
            policy_loss, value_loss = train_steps(
                candidate, optimizer, buffer, config, device, autocast_dtype, scaler
            )
            train_time = time.time() - train_start
            writer.add_scalar("loss/policy", policy_loss, iteration)
            writer.add_scalar("loss/value", value_loss, iteration)
            writer.add_scalar("time/train_sec", train_time, iteration)

            # 4. Evaluate against the current best (batched inference). Eval always
            # uses the base config (eval_simulations), never shake-boosted values.
            eval_start = time.time()
            best_net, _, _ = load_checkpoint(best_path, device)
            win_rate, wins, losses = evaluate(
                candidate, best_net, config, config.eval_games
            )
            eval_time = time.time() - eval_start
            writer.add_scalar("eval/win_rate", win_rate, iteration)
            writer.add_scalar("time/eval_sec", eval_time, iteration)
            promoted = win_rate >= config.promotion_win_rate
            # 5a. Track internal ELO; best.pt is always the strongest net so far.
            candidate_elo = controller.record_eval(win_rate, promoted)
            if promoted:
                save_checkpoint(candidate, config, iteration, best_path)
            writer.add_scalar("eval/promoted", int(promoted), iteration)

            # 5b. Iteration checkpoint (these become the bot strength levels).
            iter_path = checkpoint_dir / f"iter{iteration:03d}.pt"
            save_checkpoint(candidate, config, iteration, iter_path)

        # 6. Advance the plateau state machine (may decide to stop).
        action = controller.update()
        state.iteration = iteration
        controller.tick_time()
        state.save(state_path)

        iter_time = time.time() - started
        gpu_util = gpu.average
        writer.add_scalar("time/iteration_sec", iter_time, iteration)
        writer.add_scalar("elo/best", state.best_elo, iteration)
        writer.add_scalar("elo/candidate", candidate_elo, iteration)
        if gpu_util is not None:
            writer.add_scalar("perf/gpu_util", gpu_util, iteration)
        gpu_str = f"{gpu_util:.0f}%" if gpu_util is not None else "n/a"
        logger.info(
            "iteration %d [%s]: policy %.3f value %.3f eval %.0f%% (%dW/%dL) %s | "
            "ELO best %.0f cand %.0f (%s) | %.1f games/s, avg batch %.0f, gpu %s | "
            "self-play %.0fs, train %.0fs, eval %.0fs, total %.0fs",
            iteration, state.mode, policy_loss, value_loss, 100 * win_rate, wins, losses,
            "PROMOTED" if promoted else "kept",
            state.best_elo, candidate_elo, controller.rank_label,
            games_per_sec, sp_stats.avg_batch, gpu_str,
            sp_time, train_time, eval_time, iter_time,
        )
        if action == "stop":
            break

    final_report(config, state)
    writer.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AlphaZero training loop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(Config.load(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
