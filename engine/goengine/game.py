"""Full game state: legality (suicide, positional superko), passes, scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from goengine.board import Board
from goengine.scoring import area_score
from goengine.types import Color, IllegalMoveError, Move, Point

DEFAULT_KOMI = 7.5


@dataclass(frozen=True)
class GameResult:
    """Final result of a game.

    ``winner`` is EMPTY for a draw (impossible with fractional komi but kept
    for completeness). ``margin`` is the winning margin in points, or None
    for a resignation.
    """

    winner: Color
    margin: Optional[float]
    by_resignation: bool = False

    def __str__(self) -> str:
        if self.winner is Color.EMPTY:
            return "Draw"
        side = "B" if self.winner is Color.BLACK else "W"
        if self.by_resignation:
            return f"{side}+R"
        return f"{side}+{self.margin:g}"


class Game:
    """A Go game under Chinese rules with positional superko.

    Rules enforced:
      * No playing on occupied points.
      * No suicide.
      * Positional superko: a move may not recreate any previous whole-board
        position (regardless of side to move).
      * Two consecutive passes end the game; scoring is Chinese area scoring
        (stones + territory) with ``komi`` added to White.
    """

    def __init__(self, board_size: int = 9, komi: float = DEFAULT_KOMI) -> None:
        self.board: Board = Board(board_size)
        self.komi: float = komi
        self.current_player: Color = Color.BLACK
        self.moves: List[Move] = []
        self.captures: Dict[Color, int] = {Color.BLACK: 0, Color.WHITE: 0}
        self.consecutive_passes: int = 0
        self.result: Optional[GameResult] = None
        # All positions seen so far (including the empty board), for superko.
        self._position_history: Set[int] = {self.board.position_hash}

    @property
    def board_size(self) -> int:
        """Edge length of the board."""
        return self.board.size

    @property
    def is_over(self) -> bool:
        """Whether the game has ended (two passes or resignation)."""
        return self.result is not None

    def copy(self) -> "Game":
        """Return a deep copy of the game (cheap enough for MCTS rollouts)."""
        new = Game.__new__(Game)
        new.board = self.board.copy()
        new.komi = self.komi
        new.current_player = self.current_player
        new.moves = list(self.moves)
        new.captures = dict(self.captures)
        new.consecutive_passes = self.consecutive_passes
        new.result = self.result
        new._position_history = set(self._position_history)
        return new

    # -- legality ------------------------------------------------------------

    def _simulate(self, point: Point, color: Color) -> Tuple[Board, Set[Point]]:
        """Apply a stone placement on a copied board; raise if illegal."""
        if not self.board.in_bounds(point):
            raise IllegalMoveError(f"point {point} is off the board")
        if self.board.get(point) is not Color.EMPTY:
            raise IllegalMoveError(f"point {point} is occupied")
        trial = self.board.copy()
        captured = trial.place_stone(point, color)
        _, liberties = trial.group_at(point)
        if not liberties:
            raise IllegalMoveError(f"move at {point} is suicide")
        if trial.position_hash in self._position_history:
            raise IllegalMoveError(
                f"move at {point} violates positional superko"
            )
        return trial, captured

    def is_legal(self, move: Move) -> bool:
        """Whether ``move`` is legal for the current player."""
        if self.is_over:
            return False
        if move.is_pass or move.is_resign:
            return True
        assert move.point is not None
        try:
            self._simulate(move.point, self.current_player)
        except IllegalMoveError:
            return False
        return True

    def legal_moves(self, include_pass: bool = True) -> List[Move]:
        """All legal moves for the current player."""
        if self.is_over:
            return []
        moves = [
            Move.play(r, c)
            for (r, c) in self.board.empty_points()
            if self.is_legal(Move.play(r, c))
        ]
        if include_pass:
            moves.append(Move.pass_turn())
        return moves

    # -- playing ---------------------------------------------------------------

    def play(self, move: Move) -> None:
        """Play ``move`` for the current player, or raise IllegalMoveError."""
        if self.is_over:
            raise IllegalMoveError("game is over")
        if move.is_resign:
            self.moves.append(move)
            self.result = GameResult(
                winner=self.current_player.opponent, margin=None, by_resignation=True
            )
            return
        if move.is_pass:
            self.moves.append(move)
            self.consecutive_passes += 1
            self.current_player = self.current_player.opponent
            if self.consecutive_passes >= 2:
                self.result = self._score_result()
            return
        assert move.point is not None
        trial, captured = self._simulate(move.point, self.current_player)
        self.board = trial
        self.captures[self.current_player] += len(captured)
        self._position_history.add(self.board.position_hash)
        self.moves.append(move)
        self.consecutive_passes = 0
        self.current_player = self.current_player.opponent

    # -- scoring ---------------------------------------------------------------

    def score(self) -> Tuple[float, float]:
        """Current Chinese area score as ``(black_points, white_points)``.

        White's total includes komi. Can be called at any time, but is only
        meaningful when all dead stones have been captured (as in self-play
        games played to the end).
        """
        black_area, white_area = area_score(self.board)
        return float(black_area), float(white_area) + self.komi

    def _score_result(self) -> GameResult:
        black, white = self.score()
        if black > white:
            return GameResult(winner=Color.BLACK, margin=black - white)
        if white > black:
            return GameResult(winner=Color.WHITE, margin=white - black)
        return GameResult(winner=Color.EMPTY, margin=0.0)
