"""Chinese (area) scoring: stones on the board plus surrounded territory."""

from __future__ import annotations

from typing import Set, Tuple

from goengine.board import Board
from goengine.types import Color, Point


def area_score(board: Board) -> Tuple[int, int]:
    """Return ``(black_area, white_area)`` under area scoring.

    Area = own stones + empty regions bordered exclusively by own stones.
    Empty regions touching both colors (or touching nothing, e.g. an empty
    board) are neutral (dame) and count for nobody.
    """
    black = 0
    white = 0
    visited: Set[Point] = set()
    for point in board.points():
        state = board.get(point)
        if state is Color.BLACK:
            black += 1
        elif state is Color.WHITE:
            white += 1
        elif point not in visited:
            region, borders = _empty_region(board, point)
            visited |= region
            if borders == {Color.BLACK}:
                black += len(region)
            elif borders == {Color.WHITE}:
                white += len(region)
    return black, white


def _empty_region(board: Board, start: Point) -> Tuple[Set[Point], Set[Color]]:
    """Flood-fill the empty region containing ``start``.

    Returns the region's points and the set of stone colors adjacent to it.
    """
    region: Set[Point] = {start}
    borders: Set[Color] = set()
    frontier = [start]
    while frontier:
        p = frontier.pop()
        for n in board.neighbors(p):
            state = board.get(n)
            if state is Color.EMPTY:
                if n not in region:
                    region.add(n)
                    frontier.append(n)
            else:
                borders.add(state)
    return region, borders
