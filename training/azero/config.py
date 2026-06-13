"""Training configuration, loadable from YAML."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclasses.dataclass
class Config:
    """All hyperparameters for self-play training."""

    # Game
    board_size: int = 9
    komi: float = 7.5
    history_planes: int = 4  # number of past positions fed to the network

    # Network
    blocks: int = 6
    filters: int = 64

    # MCTS
    simulations: int = 160
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 12  # sample with T=1 for this many opening moves
    mcts_leaf_batch: int = 1  # leaves gathered per game per round; >1 uses virtual loss
    virtual_loss: float = 1.0  # discourages parallel descents from sharing a path

    # Self-play
    games_per_iteration: int = 50
    workers: int = 2
    selfplay_device: str = "cpu"  # device for the inference server; cuda -> GPU server
    selfplay_concurrent_games: int = 16  # games in flight per worker (batch multiplier)
    max_game_moves: int = 0  # 0 -> 2 * board_size^2

    # Inference server (batched GPU evaluation during self-play)
    inference_max_batch: int = 256  # fire a forward once this many positions are queued
    inference_timeout_ms: float = 5.0  # ...or after this long, whichever comes first
    precision: str = "auto"  # auto|bf16|fp16|fp32 autocast for inference + training

    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 2_000

    # Optimization
    batch_size: int = 256
    train_steps_per_iteration: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0

    # Evaluation / promotion
    eval_games: int = 20
    promotion_win_rate: float = 0.55
    eval_simulations: int = 80

    # Loop / IO
    iterations: int = 100
    device: str = "cuda"  # falls back to cpu automatically if unavailable
    run_dir: str = "runs/default"
    checkpoint_dir: str = "checkpoints"

    @property
    def move_cap(self) -> int:
        """Maximum moves per self-play game."""
        return self.max_game_moves or 2 * self.board_size * self.board_size

    @property
    def policy_size(self) -> int:
        """Board points + pass."""
        return self.board_size * self.board_size + 1

    @property
    def input_planes(self) -> int:
        """Feature planes fed to the network."""
        return 2 * self.history_planes + 1

    @staticmethod
    def load(path: str | Path) -> "Config":
        """Load a config from YAML, applying defaults for missing keys."""
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(Config)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return Config(**raw)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form (stored inside checkpoints)."""
        return dataclasses.asdict(self)
