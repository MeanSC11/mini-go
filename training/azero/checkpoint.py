"""Checkpoint save/load helpers shared by training and inference."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from azero.config import Config
from azero.network import PolicyValueNet


def _config_from_payload(raw: Dict[str, Any]) -> Config:
    """Build a Config from a checkpoint payload, ignoring keys it no longer has.

    Checkpoints embed the config used to train them; dropping unknown keys lets
    new code load older checkpoints (and vice-versa) across config schema drift.
    """
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


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
    config = _config_from_payload(payload["config"])
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


def resolve_autocast(requested: str, device: str) -> Tuple[Any, str]:
    """Pick an autocast dtype for ``device`` from a ``precision`` request.

    Returns ``(dtype, label)`` where ``dtype`` is a ``torch.dtype`` to autocast
    to, or ``None`` to run in full fp32. ``requested`` is one of
    ``auto|bf16|fp16|fp32``. Autocast only applies on CUDA; CPU always runs fp32
    here (CPU bf16 is slower than fp32 for these small convs).
    """
    if not device.startswith("cuda") or requested == "fp32":
        return None, "fp32"
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if requested == "bf16":
        return (torch.bfloat16, "bf16") if bf16_ok else (torch.float16, "fp16")
    if requested == "fp16":
        return torch.float16, "fp16"
    # auto: prefer bf16 (no GradScaler, more stable) on Ampere+/Hopper.
    if bf16_ok:
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"
