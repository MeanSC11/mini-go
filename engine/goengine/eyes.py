"""Eye detection for bot move generation.

Filling your own eye is *legal* under the rules, so this lives outside the rule
engine — but every bot needs it: without it a bot keeps playing pointless moves
(even filling its own eyes) until the board is full instead of passing once the
game is decided. Excluding own-eye fills from the candidate moves means that when
only such points remain, ``pass`` is the only candidate, both sides pass, and the
game ends for scoring.
"""

from __future__ import annotations

from typing import List

from goengine.board import Board
from goengine.types import Color, Move, Point


def is_eye_fill(board: Board, point: Point, color: Color) -> bool:
    """Whether playing ``color`` at empty ``point`` would fill its own eye.

    True when every orthogonal neighbour is ``color`` (board edges act as
    friendly walls) and the diagonals are controlled: at an edge/corner all
    on-board diagonals must be ``color``; in the interior at most one may be the
    opponent's. Such a move has no opponent stone adjacent, so it can never
    capture — excluding it from a bot's candidates is always safe.
    """
    if board.get(point) is not Color.EMPTY:
        return False
    for neighbor in board.neighbors(point):
        if board.get(neighbor) is not color:
            return False
    r, c = point
    diagonals = [(r + dr, c + dc) for dr in (-1, 1) for dc in (-1, 1)]
    on_board = [d for d in diagonals if board.in_bounds(d)]
    enemy = sum(1 for d in on_board if board.get(d) is color.opponent)
    if len(on_board) < 4:  # edge or corner: any enemy diagonal makes it a false eye
        return enemy == 0
    return enemy <= 1  # interior: tolerate a single enemy diagonal


def candidate_moves(game, include_pass: bool = True) -> List[Move]:
    """Legal moves for the side to move, excluding own-eye fills (+ optional pass).

    This is what bots should search over instead of :meth:`Game.legal_moves`, so
    a finished side passes rather than filling its own eyes.
    """
    color = game.current_player
    moves = [
        Move.play(r, c)
        for (r, c) in game.board.empty_points()
        if not is_eye_fill(game.board, (r, c), color)
        and game.is_legal(Move.play(r, c))
    ]
    if include_pass:
        moves.append(Move.pass_turn())
    return moves
