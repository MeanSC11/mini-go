"""Plateau-aware training driver: track internal ELO and escalate before stopping.

The training loop reports, every iteration, the win rate of the candidate vs the
current best. We turn that into an internal ELO (anchored at 0 for the iter-0
random net) and run a three-layer stopping policy:

  1. **Detect**  -- if best ELO hasn't risen by more than ``plateau_elo_threshold``
     for ``plateau_patience`` iterations, suspect a plateau.
  2. **Shake**   -- don't quit yet: lower the learning rate, raise self-play
     exploration (Dirichlet noise + opening temperature) and simulations, and run
     ``shake_iterations`` more. If ELO climbs again, go back to normal.
  3. **Accept**  -- if the shake didn't help, this network + compute has topped
     out; stop and report the ceiling.

A wall-clock ``max_hours`` circuit breaker stops the run regardless, and the full
state is persisted every iteration so a dropped pod can resume exactly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

from azero import rank as rank_mod
from azero.config import Config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingState:
    """Everything needed to resume the driver after a restart."""

    best_elo: float = 0.0
    candidate_elo: float = 0.0
    plateau_anchor_elo: float = 0.0  # ELO we measure progress against
    stale_count: int = 0
    mode: str = "normal"  # "normal" | "shake"
    shake_remaining: int = 0
    shake_attempts: int = 0
    iteration: int = 0
    elapsed_seconds: float = 0.0  # accumulated wall time across runs
    stop_reason: str = ""

    @classmethod
    def load(cls, path: Path) -> "TrainingState":
        if Path(path).is_file():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            known = {f.name for f in dataclasses.fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        return cls()

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")


class PlateauController:
    """Owns ELO bookkeeping, shake escalation, and the stop decision."""

    def __init__(self, config: Config, state: TrainingState) -> None:
        self.config = config
        self.state = state
        self._base_lr = config.learning_rate
        self._run_start = time.monotonic()
        self._elapsed_at_start = state.elapsed_seconds

    # -- wall-clock ---------------------------------------------------------

    def elapsed_hours(self) -> float:
        return self.current_elapsed_seconds() / 3600.0

    def current_elapsed_seconds(self) -> float:
        return self._elapsed_at_start + (time.monotonic() - self._run_start)

    def tick_time(self) -> None:
        self.state.elapsed_seconds = self.current_elapsed_seconds()

    # -- shake-adjusted knobs ----------------------------------------------

    def effective_selfplay_config(self) -> Config:
        """Config used for self-play this iteration (boosted while shaking)."""
        if self.state.mode != "shake":
            return self.config
        c = self.config
        return dataclasses.replace(
            c,
            simulations=max(1, int(round(c.simulations * c.shake_sim_multiplier))),
            dirichlet_eps=c.shake_dirichlet_eps,
            temperature_moves=c.temperature_moves + c.shake_temperature_moves_bonus,
        )

    def current_lr(self) -> float:
        return self._base_lr * (self.config.shake_lr_factor if self.state.mode == "shake" else 1.0)

    @property
    def rank_label(self) -> str:
        return rank_mod.elo_to_label(self.state.best_elo)

    # -- ELO + plateau state machine ---------------------------------------

    def record_eval(self, win_rate: float, promoted: bool) -> float:
        """Update ELO from a candidate-vs-best result; return candidate ELO."""
        candidate = self.state.best_elo + rank_mod.winrate_to_elo_diff(win_rate)
        self.state.candidate_elo = candidate
        if promoted:
            self.state.best_elo = max(self.state.best_elo, candidate)
        return candidate

    def update(self) -> str:
        """Advance the state machine after an eval. Returns 'continue' or 'stop'."""
        gain = self.state.best_elo - self.state.plateau_anchor_elo
        improved = gain > self.config.plateau_elo_threshold

        if self.state.mode == "normal":
            if improved:
                self.state.plateau_anchor_elo = self.state.best_elo
                self.state.stale_count = 0
            else:
                self.state.stale_count += 1
                if (self.config.stop_on_plateau
                        and self.state.stale_count >= self.config.plateau_patience):
                    self._enter_shake()
            return "continue"

        # shake mode
        if improved:
            logger.info(
                "SHAKE worked: best ELO %.0f (+%.0f) — resuming NORMAL training",
                self.state.best_elo, gain,
            )
            self.state.mode = "normal"
            self.state.plateau_anchor_elo = self.state.best_elo
            self.state.stale_count = 0
            return "continue"

        self.state.shake_remaining -= 1
        if self.state.shake_remaining <= 0:
            self.state.stop_reason = "plateau"
            logger.info("SHAKE #%d did not break the plateau — ceiling reached",
                        self.state.shake_attempts)
            return "stop"
        return "continue"

    def _enter_shake(self) -> None:
        c = self.config
        self.state.mode = "shake"
        self.state.shake_remaining = c.shake_iterations
        self.state.shake_attempts += 1
        self.state.plateau_anchor_elo = self.state.best_elo
        self.state.stale_count = 0
        logger.info(
            "PLATEAU suspected (no >%.0f ELO gain for %d iters) — entering SHAKE #%d: "
            "lr x%.2f, dirichlet_eps %.2f, temp_moves +%d, sims x%.1f for %d iters",
            c.plateau_elo_threshold, c.plateau_patience, self.state.shake_attempts,
            c.shake_lr_factor, c.shake_dirichlet_eps, c.shake_temperature_moves_bonus,
            c.shake_sim_multiplier, c.shake_iterations,
        )


def final_report(config: Config, state: TrainingState) -> None:
    """Log the stopping summary and a 'to go further' assessment."""
    reason = state.stop_reason or "iterations exhausted"
    logger.info("=" * 68)
    if reason == "plateau":
        logger.info(
            "STOPPED: plateau — shook %d time(s), best ELO did not rise > %.0f after.",
            state.shake_attempts, config.plateau_elo_threshold,
        )
    elif reason == "max_hours":
        logger.info("STOPPED: max_hours circuit breaker (%.1f h).", config.max_hours)
    else:
        logger.info("STOPPED: %s.", reason)
    logger.info(
        "Best internal ELO %.0f  ->  rank estimate %s (uncalibrated; "
        "run azero-benchmark vs a reference engine for an accurate rank).",
        state.best_elo, rank_mod.elo_to_label(state.best_elo),
    )
    logger.info(
        "Reached in %d iterations / %.1f h. best.pt holds this strongest network.",
        state.iteration, state.elapsed_seconds / 3600.0,
    )
    logger.info(
        "Ceiling assessment: a %d-block x %d-filter net on %dx%d at %d self-play sims "
        "topped out here. To go stronger: (1) bigger network (more blocks/filters), "
        "(2) more compute (more games/iteration, more simulations, longer training), "
        "(3) distributed self-play to raise games/sec. Serve strength can still be "
        "pushed now by giving more simulations/time per move (see azero-assess).",
        config.blocks, config.filters, config.board_size, config.board_size,
        config.simulations,
    )
    logger.info("=" * 68)
