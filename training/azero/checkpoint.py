"""Checkpoint save/load helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from azero.config import Config
from azero.network import PolicyValueNet


def save_checkpoint(
    net: PolicyValueNet, config: Config, iteration: int, path: str | Path
) -> None:
    """Persist model weights plus the architecture info needed to rebuild it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "config": config.to_dict(),
            "iteration": iteration,
        },
        str(path),
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> Tuple[PolicyValueNet, Config, int]:
    """Rebuild the network from a checkpoint file."""
    payload: Dict[str, Any] = torch.load(
        str(path), map_location=device, weights_only=False
    )
    config = Config(**payload["config"])
    net = PolicyValueNet(
        board_size=config.board_size,
        input_planes=config.input_planes,
        blocks=config.blocks,
        filters=config.filters,
    ).to(device)
    net.load_state_dict(payload["model"])
    net.eval()
    return net, config, int(payload.get("iteration", 0))


def build_network(config: Config, device: str) -> PolicyValueNet:
    """Fresh network matching ``config``."""
    return PolicyValueNet(
        board_size=config.board_size,
        input_planes=config.input_planes,
        blocks=config.blocks,
        filters=config.filters,
    ).to(device)


def resolve_device(requested: str) -> str:
    """Use CUDA only when actually available."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested
