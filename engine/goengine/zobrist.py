"""Deterministic Zobrist hashing for positional-superko detection.

Hashes depend only on stone placement (not side to move), which is what
positional superko requires. Tables are generated lazily per board size with
a fixed seed so hashes are reproducible across processes.
"""

from __future__ import annotations

import random
from typing import Dict, List

from goengine.types import Color

_SEED = 0x60_BAD0C
_tables: Dict[int, List[List[int]]] = {}


def table(size: int) -> List[List[int]]:
    """Return the Zobrist table for ``size``: ``table[point_index][color]``.

    ``color`` indexes are ``Color.BLACK`` and ``Color.WHITE``; index 0 is
    unused (empty points do not contribute to the hash).
    """
    cached = _tables.get(size)
    if cached is None:
        rng = random.Random(_SEED + size)
        cached = [
            [0, rng.getrandbits(64), rng.getrandbits(64)]
            for _ in range(size * size)
        ]
        _tables[size] = cached
    return cached


def stone_hash(size: int, row: int, col: int, color: Color) -> int:
    """Hash contribution of a single stone; XOR to add or remove it."""
    return table(size)[row * size + col][color]
