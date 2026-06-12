"""FastAPI inference service: turns positions into bot moves/analysis."""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from goengine import Color, Game, IllegalMoveError, Move

from app.registry import available_levels, get_bot

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="go-bot inference service")


class MovePayload(BaseModel):
    """Wire format for a move (matches the game server)."""

    type: Literal["play", "pass", "resign"]
    row: Optional[int] = None
    col: Optional[int] = None


class PositionRequest(BaseModel):
    """A position described by its move history."""

    board_size: int = Field(default=9, ge=5, le=19)
    komi: float = 7.5
    moves: List[MovePayload] = []
    level: str = "random"


class MoveResponse(BaseModel):
    """The bot's move plus optional analysis."""

    move: MovePayload
    win_rate: Optional[float] = None  # black's win probability
    policy: Dict[str, float] = {}


def replay(request: PositionRequest) -> Game:
    """Rebuild the game from its move list; the engine validates each move."""
    game = Game(board_size=request.board_size, komi=request.komi)
    try:
        for payload in request.moves:
            if payload.type == "play":
                if payload.row is None or payload.col is None:
                    raise IllegalMoveError("play move missing coordinates")
                game.play(Move.play(payload.row, payload.col))
            elif payload.type == "pass":
                game.play(Move.pass_turn())
            else:
                game.play(Move.resign())
    except IllegalMoveError as exc:
        raise HTTPException(status_code=422, detail=f"invalid history: {exc}") from exc
    return game


def black_win_rate(game: Game, side_to_move_rate: Optional[float]) -> Optional[float]:
    """Convert a side-to-move win rate into black's perspective."""
    if side_to_move_rate is None:
        return None
    if game.current_player is Color.BLACK:
        return side_to_move_rate
    return 1.0 - side_to_move_rate


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/levels")
async def levels() -> dict:
    """All playable bot levels."""
    return {"levels": available_levels()}


@app.post("/move", response_model=MoveResponse)
async def choose_move(request: PositionRequest) -> MoveResponse:
    """Choose a move for the side to play in the given position."""
    game = replay(request)
    if game.is_over:
        raise HTTPException(status_code=422, detail="game is already over")
    bot = get_bot(request.level)
    move, rate, policy = bot.search(game)
    if move.is_pass:
        payload = MovePayload(type="pass")
    else:
        assert move.point is not None
        payload = MovePayload(type="play", row=move.point[0], col=move.point[1])
    return MoveResponse(
        move=payload, win_rate=black_win_rate(game, rate), policy=policy
    )


@app.post("/analyze", response_model=MoveResponse)
async def analyze(request: PositionRequest) -> MoveResponse:
    """Analyze a position without committing to a move (same payload)."""
    return await choose_move(request)
