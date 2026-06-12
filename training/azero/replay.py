"""Replay buffer of self-play samples."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np

Sample = Tuple[np.ndarray, np.ndarray, float]
"""(input planes, policy target, outcome z for the player to move)."""


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform sampling and disk persistence."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: Deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, samples: List[Sample]) -> None:
        """Append new samples (oldest are evicted past capacity)."""
        self._data.extend(samples)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a training batch as stacked arrays."""
        batch = random.sample(self._data, min(batch_size, len(self._data)))
        states = np.stack([s[0] for s in batch])
        policies = np.stack([s[1] for s in batch])
        values = np.asarray([s[2] for s in batch], dtype=np.float32)
        return states, policies, values

    def save(self, path: str | Path) -> None:
        """Persist the buffer to a compressed .npz file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        states = np.stack([s[0] for s in self._data]) if self._data else np.empty(0)
        policies = np.stack([s[1] for s in self._data]) if self._data else np.empty(0)
        values = np.asarray([s[2] for s in self._data], dtype=np.float32)
        np.savez_compressed(str(path), states=states, policies=policies, values=values)

    def load(self, path: str | Path) -> None:
        """Restore a buffer previously written by :meth:`save`."""
        archive = np.load(str(path))
        states, policies, values = (
            archive["states"],
            archive["policies"],
            archive["values"],
        )
        self._data.clear()
        for i in range(len(values)):
            self._data.append((states[i], policies[i], float(values[i])))
