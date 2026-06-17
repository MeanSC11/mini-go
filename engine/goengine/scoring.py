"""Chinese (area) scoring with dead-stone removal by simulated play.

Scoring a finished game is two steps:

1. **Remove dead stones.** A group that is enclosed by a single opponent color
   is tested for life. If it clearly has two eyes (or enough enclosed space to
   make them) it lives; otherwise its life is resolved by *simulated play* --
   a local playout inside the enclosed region with the opponent moving first.
   If the group is captured in the majority of playouts it is dead and removed,
   and the space it occupied becomes the enclosing player's territory. This is
   the step the previous implementation lacked: a surrounded-but-not-lifted
   group used to be counted as alive, scoring points for its (already lost)
   owner.

2. **Area-score the cleaned board (Tromp-Taylor):** each player scores their
   remaining stones plus empty regions bordered only by their color.

Only groups *enclosed by the opponent* are candidates for removal, so open
frameworks bordering their own territory (and the edge) are never touched --
the danger with naive dead-stone heuristics is removing live stones, and this
avoids it. Life of large enclosed groups is decided by their eye space; only
small/ambiguous shapes pay for a playout, so this stays cheap on a played-out
board (the enclosed empty pockets are small).
"""

from __future__ import annotations

import random
from typing import List, Set, Tuple

from goengine.board import Board
from goengine.types import Color, Point

# A single enclosed empty region this large can always make two eyes -> alive.
_ALIVE_EYE_SPACE = 6
# If a group can reach more enclosed empty space than this, it is not in a tight
# pocket -- treat it as alive rather than risk removing a live group (and avoid
# huge, unstable playouts on open/unsettled boards).
_MAX_DEAD_POCKET = 14
_DEFAULT_PLAYOUTS = 9


def area_score(board: Board) -> Tuple[int, int]:
    """Tromp-Taylor area of the board as-is (no dead-stone removal).

    Used for already-resolved boards (e.g. the tail of an MCTS rollout). For
    scoring a finished game use :func:`score_position`, which removes dead
    stones first.
    """
    return _tromp_taylor_area(board)


def score_position(board: Board, playouts: int = _DEFAULT_PLAYOUTS) -> Tuple[int, int]:
    """Return ``(black_area, white_area)`` after removing dead stones.

    Operates on a copy; the input board is not modified.
    """
    work = board.copy()
    removed: Set[Point] = set()
    for color in (Color.BLACK, Color.WHITE):
        opponent = color.opponent
        visited: Set[Point] = set()
        for start in board.points():
            if board.get(start) is not color or start in visited:
                continue
            stones, empties, border = _enclosed_cluster(board, start, color)
            visited |= stones
            if border != {opponent}:
                continue  # not enclosed purely by the opponent: leave it alone
            if not _cluster_is_alive(board, stones, empties, color, opponent, playouts):
                removed |= stones
    for p in removed:
        work._set(p, Color.EMPTY)
    return _tromp_taylor_area(work)


# -- Tromp-Taylor area -------------------------------------------------------


def _tromp_taylor_area(board: Board) -> Tuple[int, int]:
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
    """Flood-fill the empty region containing ``start``; return points + borders."""
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


# -- enclosure & life --------------------------------------------------------


def _enclosed_cluster(
    board: Board, start: Point, color: Color
) -> Tuple[Set[Point], Set[Point], Set[Color]]:
    """Flood from ``start`` over ``color`` stones and the empties they reach.

    Returns ``(stones, empties, border)`` where ``border`` is the set of
    *other* stone colors adjacent to the cluster. Friendly groups connected
    through shared empty space are merged into one cluster, so they are judged
    for life together.
    """
    stones: Set[Point] = set()
    empties: Set[Point] = set()
    border: Set[Color] = set()
    stack = [start]
    while stack:
        p = stack.pop()
        value = board.get(p)
        if value is color:
            if p in stones:
                continue
            stones.add(p)
        elif value is Color.EMPTY:
            if p in empties:
                continue
            empties.add(p)
        for n in board.neighbors(p):
            nv = board.get(n)
            if nv is color:
                if n not in stones:
                    stack.append(n)
            elif nv is Color.EMPTY:
                if n not in empties:
                    stack.append(n)
            else:
                border.add(nv)
    return stones, empties, border


def _real_eye_regions(
    board: Board, stones: Set[Point], empties: Set[Point], color: Color
) -> List[Set[Point]]:
    """Connected empty regions (within ``empties``) bordered only by ``stones``."""
    regions: List[Set[Point]] = []
    seen: Set[Point] = set()
    for start in empties:
        if start in seen:
            continue
        component: Set[Point] = set()
        bordered_only_by_cluster = True
        stack = [start]
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            for n in board.neighbors(p):
                nv = board.get(n)
                if nv is Color.EMPTY:
                    if n in empties and n not in component:
                        stack.append(n)
                elif n not in stones:
                    bordered_only_by_cluster = False
        seen |= component
        if bordered_only_by_cluster:
            regions.append(component)
    return regions


def _cluster_is_alive(
    board: Board,
    stones: Set[Point],
    empties: Set[Point],
    color: Color,
    opponent: Color,
    playouts: int,
) -> bool:
    """Whether an opponent-enclosed cluster lives."""
    if not empties:
        return False  # no liberties at all (degenerate); treat as dead
    eyes = _real_eye_regions(board, stones, empties, color)
    if len(eyes) >= 2:
        return True
    if any(len(region) >= _ALIVE_EYE_SPACE for region in eyes):
        return True
    if len(empties) > _MAX_DEAD_POCKET:
        return True  # not a tight pocket: enough room to live, don't remove
    anchor = min(stones)
    survivals = 0
    for k in range(playouts):
        seed = board.position_hash + 9176 * k + anchor[0] * 31 + anchor[1]
        if _local_playout_survives(
            board, stones, empties, color, opponent, random.Random(seed)
        ):
            survivals += 1
    return survivals * 2 > playouts


def _local_playout_survives(
    board: Board,
    stones: Set[Point],
    empties: Set[Point],
    color: Color,
    opponent: Color,
    rng: random.Random,
) -> bool:
    """Play out the enclosed region (opponent first); does any cluster stone live?"""
    work = board.copy()
    region = set(empties)
    target = set(stones)
    to_move = opponent  # the attacker tries to capture; defender tries to live
    passes = 0
    move_limit = len(region) * 4 + 10
    for _ in range(move_limit):
        if passes >= 2:
            break
        candidates = [
            p
            for p in region
            if work.get(p) is Color.EMPTY
            and _legal_for_playout(work, p, to_move)
            and not is_simple_eye(work, p, to_move)
        ]
        if not candidates:
            passes += 1
        else:
            passes = 0
            work.place_stone(rng.choice(candidates), to_move)
            if all(work.get(s) is not color for s in target):
                return False  # whole cluster captured
        to_move = to_move.opponent
    return any(work.get(s) is color for s in target)


def _legal_for_playout(board: Board, point: Point, color: Color) -> bool:
    """Cheap non-suicide check (a capturing move is legal)."""
    opponent = color.opponent
    for n in board.neighbors(point):
        value = board.get(n)
        if value is Color.EMPTY:
            return True
        if value is color:
            _, liberties = board.group_at(n)
            if len(liberties) > 1:
                return True
        elif value is opponent:
            _, liberties = board.group_at(n)
            if len(liberties) == 1:
                return True  # captures the opponent group
    return False


def is_simple_eye(board: Board, point: Point, color: Color) -> bool:
    """Heuristic single-point eye check for ``color`` at an empty ``point``."""
    for n in board.neighbors(point):
        if board.get(n) is not color:
            return False
    r, c = point
    diagonals = [
        (r + dr, c + dc)
        for dr in (-1, 1)
        for dc in (-1, 1)
        if board.in_bounds((r + dr, c + dc))
    ]
    enemy = sum(1 for d in diagonals if board.get(d) is color.opponent)
    if len(diagonals) < 4:  # edge or corner
        return enemy == 0
    return enemy <= 1
