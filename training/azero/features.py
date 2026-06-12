"""Board -> network input encoding.

Input tensor layout (``2 * H + 1`` planes of ``size x size``):
  * planes ``0..H-1``  : current player's stones at t, t-1, ..., t-H+1
  * planes ``H..2H-1`` : opponent's stones at t, t-1, ..., t-H+1
  * plane  ``2H``      : all ones if black is to move, else all zeros
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from goengine import Board, Color, Game, Move


class HistoryTracker:
    """Keeps the last ``history`` board snapshots of a game."""

    def __init__(self, history: int, initial_board: Optional[Board] = None) -> None:
        self.history = history
        self._boards: Deque[Board] = deque(maxlen=history)
        if initial_board is not None:
            self.push(initial_board)

    def push(self, board: Board) -> None:
        """Record a snapshot (copied) after a move was applied."""
        self._boards.append(board.copy())

    def boards(self) -> List[Board]:
        """Snapshots, most recent first."""
        return list(reversed(self._boards))


def encode(
    current: Board, tracker: Optional[HistoryTracker], to_play: Color, history: int
) -> np.ndarray:
    """Encode a position into network input planes (float32).

    ``tracker`` may be None, in which case only the current board is used and
    older history planes are zero.
    """
    size = current.size
    planes = np.zeros((2 * history + 1, size, size), dtype=np.float32)
    boards: List[Board] = tracker.boards() if tracker is not None else [current]
    opponent = to_play.opponent
    for t, board in enumerate(boards[:history]):
        for r in range(size):
            for c in range(size):
                stone = board.get((r, c))
                if stone is to_play:
                    planes[t, r, c] = 1.0
                elif stone is opponent:
                    planes[history + t, r, c] = 1.0
    if to_play is Color.BLACK:
        planes[2 * history].fill(1.0)
    return planes


def encode_game(game: Game, history: int) -> np.ndarray:
    """Encode ``game``'s current position by replaying it for history planes."""
    replay = Game(game.board_size, game.komi)
    tracker = HistoryTracker(history, replay.board)
    for move in game.moves:
        if move.is_resign:
            break
        replay.play(move)
        tracker.push(replay.board)
    return encode(replay.board, tracker, replay.current_player, history)


def move_to_index(move: Move, size: int) -> int:
    """Map a move to a policy index (pass is the last index)."""
    if move.is_pass:
        return size * size
    assert move.point is not None
    return move.point[0] * size + move.point[1]


def index_to_move(index: int, size: int) -> Move:
    """Inverse of :func:`move_to_index`."""
    if index == size * size:
        return Move.pass_turn()
    return Move.play(index // size, index % size)
