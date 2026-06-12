"""Low-level board: stone placement, group/liberty tracking, captures."""

from __future__ import annotations

from typing import Iterator, List, Set, Tuple

from goengine import zobrist
from goengine.types import Color, Point


class Board:
    """A Go board of arbitrary square size.

    Tracks stone placement and an incrementally-updated Zobrist hash of the
    position (stones only, no side-to-move), used for positional superko.
    """

    def __init__(self, size: int = 9) -> None:
        if size < 2 or size > 25:
            raise ValueError(f"unsupported board size: {size}")
        self.size: int = size
        self._grid: List[Color] = [Color.EMPTY] * (size * size)
        self.position_hash: int = 0

    def copy(self) -> "Board":
        """Return a deep copy of this board."""
        new = Board.__new__(Board)
        new.size = self.size
        new._grid = list(self._grid)
        new.position_hash = self.position_hash
        return new

    # -- point access ------------------------------------------------------

    def in_bounds(self, point: Point) -> bool:
        """Whether ``point`` lies on the board."""
        r, c = point
        return 0 <= r < self.size and 0 <= c < self.size

    def get(self, point: Point) -> Color:
        """Color at ``point`` (EMPTY if unoccupied)."""
        r, c = point
        return self._grid[r * self.size + c]

    def _set(self, point: Point, color: Color) -> None:
        """Set ``point`` to ``color``, keeping the Zobrist hash in sync."""
        r, c = point
        idx = r * self.size + c
        old = self._grid[idx]
        if old is not Color.EMPTY:
            self.position_hash ^= zobrist.stone_hash(self.size, r, c, old)
        if color is not Color.EMPTY:
            self.position_hash ^= zobrist.stone_hash(self.size, r, c, color)
        self._grid[idx] = color

    def neighbors(self, point: Point) -> Iterator[Point]:
        """Yield orthogonal neighbors of ``point`` that are on the board."""
        r, c = point
        if r > 0:
            yield (r - 1, c)
        if r < self.size - 1:
            yield (r + 1, c)
        if c > 0:
            yield (r, c - 1)
        if c < self.size - 1:
            yield (r, c + 1)

    def points(self) -> Iterator[Point]:
        """Yield every point on the board in row-major order."""
        for r in range(self.size):
            for c in range(self.size):
                yield (r, c)

    def empty_points(self) -> Iterator[Point]:
        """Yield every empty point."""
        for p in self.points():
            if self.get(p) is Color.EMPTY:
                yield p

    # -- groups ------------------------------------------------------------

    def group_at(self, point: Point) -> Tuple[Set[Point], Set[Point]]:
        """Return ``(stones, liberties)`` of the group containing ``point``.

        ``point`` must be occupied.
        """
        color = self.get(point)
        if color is Color.EMPTY:
            raise ValueError(f"no stone at {point}")
        stones: Set[Point] = {point}
        liberties: Set[Point] = set()
        frontier = [point]
        while frontier:
            p = frontier.pop()
            for n in self.neighbors(p):
                state = self.get(n)
                if state is Color.EMPTY:
                    liberties.add(n)
                elif state is color and n not in stones:
                    stones.add(n)
                    frontier.append(n)
        return stones, liberties

    # -- move application --------------------------------------------------

    def place_stone(self, point: Point, color: Color) -> Set[Point]:
        """Place ``color`` at ``point``, removing any captured opponent groups.

        Returns the set of captured opponent stones. Does NOT validate
        suicide or ko — that is the responsibility of :class:`goengine.game.Game`,
        which simulates moves on a copy first.
        """
        if self.get(point) is not Color.EMPTY:
            raise ValueError(f"point {point} is occupied")
        self._set(point, color)
        captured: Set[Point] = set()
        opponent = color.opponent
        for n in self.neighbors(point):
            if self.get(n) is opponent:
                stones, liberties = self.group_at(n)
                if not liberties:
                    captured |= stones
        for p in captured:
            self._set(p, Color.EMPTY)
        return captured

    # -- rendering ---------------------------------------------------------

    def __str__(self) -> str:
        symbols = {Color.EMPTY: ".", Color.BLACK: "X", Color.WHITE: "O"}
        rows = []
        for r in range(self.size):
            rows.append(" ".join(symbols[self.get((r, c))] for c in range(self.size)))
        return "\n".join(rows)
