"""Residual policy+value network (AlphaZero-style)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and a skip connection."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class PolicyValueNet(nn.Module):
    """ResNet trunk with a policy head (board+pass logits) and a value head."""

    def __init__(
        self,
        board_size: int = 9,
        input_planes: int = 9,
        blocks: int = 6,
        filters: int = 64,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.input_planes = input_planes
        self.policy_size = board_size * board_size + 1

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(filters) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value)``; value is in [-1, 1]."""
        features = self.trunk(self.stem(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def predict(self, planes: "torch.Tensor") -> Tuple[torch.Tensor, float]:
        """Single-position inference: softmax policy and scalar value."""
        self.eval()
        x = planes.unsqueeze(0).to(next(self.parameters()).device)
        logits, value = self.forward(x)
        return F.softmax(logits[0], dim=0).cpu(), float(value.item())

    @torch.no_grad()
    def predict_many(
        self, planes: "np.ndarray", autocast_dtype=None
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Batched inference: evaluate ``N`` positions in one forward pass.

        ``planes`` is ``(N, C, H, W)`` float32. Returns ``(policies, values)``
        as numpy arrays of shape ``(N, policy_size)`` and ``(N,)``. This is the
        path that keeps a GPU busy during self-play — one big batch instead of
        ``N`` separate batch-1 calls. ``autocast_dtype`` (e.g. ``torch.bfloat16``)
        enables mixed-precision inference on CUDA.
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(planes).to(device, non_blocking=True)
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, value = self.forward(x)
        else:
            logits, value = self.forward(x)
        policies = F.softmax(logits.float(), dim=1).cpu().numpy()
        values = value.float().reshape(-1).cpu().numpy()
        return policies, values
