"""Pydantic request/response schemas and wire-format helpers."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from goengine import Color, Game, Move


class MovePayload(BaseModel):
    """A move intent from a client. The server validates it via the engine."""

    type: Literal["play", "pass", "resign"]
    row: Optional[int] = None
    col: Optional[int] = None

    def to_move(self) -> Move:
        """Convert to an engine move."""
        if self.type == "play":
            if self.row is None or self.col is None:
                raise ValueError("play move requires row and col")
            return Move.play(self.row, self.col)
        if self.type == "pass":
            return Move.pass_turn()
        return Move.resign()

    @staticmethod
    def from_move(move: Move) -> "MovePayload":
        """Convert an engine move to the wire format."""
        if move.is_resign:
            return MovePayload(type="resign")
        if move.is_pass:
            return MovePayload(type="pass")
        assert move.point is not None
        return MovePayload(type="play", row=move.point[0], col=move.point[1])


class CreateGameRequest(BaseModel):
    """Create a new game."""

    board_size: int = Field(default=9, ge=5, le=19)
    komi: float = 7.5
    mode: Literal["hvh", "hvb"] = "hvh"
    bot_level: Optional[str] = None
    user_id: Optional[str] = None
    creator_color: Literal["black", "white"] = "black"


class JoinGameRequest(BaseModel):
    """Join an existing waiting game."""

    user_id: Optional[str] = None


class PlayerCredentials(BaseModel):
    """Returned to a player on create/join; the token authorizes moves."""

    game_id: str
    color: Literal["black", "white"]
    token: str


class GameStateResponse(BaseModel):
    """Full snapshot of a game, safe to send to any client."""

    game_id: str
    board_size: int
    komi: float
    mode: str
    bot_level: Optional[str]
    status: str
    board: List[str]
    current_player: Literal["black", "white"]
    captures_black: int
    captures_white: int
    moves: List[MovePayload]
    result: Optional[str]
    score_black: Optional[float]
    score_white: Optional[float]


class CreateUserRequest(BaseModel):
    """Register a (display-name only) user."""

    name: str = Field(min_length=1, max_length=64)


class UserResponse(BaseModel):
    """Public user info."""

    id: str
    name: str
    elo: float
    games_played: int


def color_name(color: Color) -> Literal["black", "white"]:
    """Engine color -> wire string."""
    return "black" if color is Color.BLACK else "white"


def board_rows(game: Game) -> List[str]:
    """Serialize the board as strings of '.', 'B', 'W' per row."""
    symbols = {Color.EMPTY: ".", Color.BLACK: "B", Color.WHITE: "W"}
    return [
        "".join(symbols[game.board.get((r, c))] for c in range(game.board_size))
        for r in range(game.board_size)
    ]
