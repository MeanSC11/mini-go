"""Core value types shared across the engine."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[int, int]
"""Board coordinate as ``(row, col)``, zero-indexed from the top-left."""


class Color(enum.IntEnum):
    """Stone color / point state."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Color":
        """Return the opposing color. Only valid for BLACK/WHITE."""
        if self is Color.BLACK:
            return Color.WHITE
        if self is Color.WHITE:
            return Color.BLACK
        raise ValueError("EMPTY has no opponent")


class IllegalMoveError(Exception):
    """Raised when a move violates the rules."""


@dataclass(frozen=True)
class Move:
    """A move: either placing a stone at ``point`` or a pass.

    Use :meth:`play` / :meth:`pass_turn` constructors instead of building
    instances directly.
    """

    point: Optional[Point] = None
    is_pass: bool = False
    is_resign: bool = False

    @staticmethod
    def play(row: int, col: int) -> "Move":
        """Create a stone-placing move at ``(row, col)``."""
        return Move(point=(row, col))

    @staticmethod
    def pass_turn() -> "Move":
        """Create a pass move."""
        return Move(is_pass=True)

    @staticmethod
    def resign() -> "Move":
        """Create a resignation."""
        return Move(is_resign=True)
