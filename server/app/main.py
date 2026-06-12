"""FastAPI application: REST API + WebSocket for real-time play."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from goengine import Color, Game, IllegalMoveError, game_to_sgf

from app.bots import list_bot_levels, request_analysis
from app.config import settings
from app.db import GameRow, UserRow, get_session, init_db
from app.schemas import (
    CreateGameRequest,
    CreateUserRequest,
    GameStateResponse,
    JoinGameRequest,
    MovePayload,
    PlayerCredentials,
    UserResponse,
    board_rows,
)
from app.sessions import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create database tables on startup."""
    await init_db()
    yield


app = FastAPI(title="go-bot server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


# -- users -----------------------------------------------------------------


@app.post("/api/users", response_model=UserResponse)
async def create_user(
    body: CreateUserRequest, db: AsyncSession = Depends(get_session)
) -> UserResponse:
    """Register a user (or return the existing one with that name)."""
    existing = await db.scalar(select(UserRow).where(UserRow.name == body.name))
    if existing is not None:
        return UserResponse(
            id=existing.id,
            name=existing.name,
            elo=existing.elo,
            games_played=existing.games_played,
        )
    user = UserRow(name=body.name)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserResponse(
        id=user.id, name=user.name, elo=user.elo, games_played=user.games_played
    )


@app.get("/api/users", response_model=List[UserResponse])
async def leaderboard(db: AsyncSession = Depends(get_session)) -> List[UserResponse]:
    """Top players by ELO."""
    rows = (
        await db.scalars(select(UserRow).order_by(UserRow.elo.desc()).limit(50))
    ).all()
    return [
        UserResponse(id=u.id, name=u.name, elo=u.elo, games_played=u.games_played)
        for u in rows
    ]


# -- games ------------------------------------------------------------------


@app.post("/api/games", response_model=PlayerCredentials)
async def create_game(
    body: CreateGameRequest, db: AsyncSession = Depends(get_session)
) -> PlayerCredentials:
    """Create a game and seat the creator."""
    creator_color = Color.BLACK if body.creator_color == "black" else Color.WHITE
    bot_color = creator_color.opponent if body.mode == "hvb" else Color.WHITE
    session = manager.create(
        board_size=body.board_size,
        komi=body.komi,
        mode=body.mode,
        bot_level=body.bot_level or settings.bot_default_level,
        bot_color=bot_color,
    )
    token = session.add_player(creator_color, body.user_id)
    if body.mode == "hvb":
        session.status = "playing"
    row = GameRow(
        id=session.id,
        board_size=body.board_size,
        komi=body.komi,
        mode=body.mode,
        bot_level=session.bot_level,
        status=session.status,
        sgf=game_to_sgf(session.game),
    )
    if creator_color is Color.BLACK:
        row.black_user_id = body.user_id
    else:
        row.white_user_id = body.user_id
    db.add(row)
    await db.commit()
    # If the bot plays black, kick off its first move.
    session.maybe_start_bot_turn()
    return PlayerCredentials(
        game_id=session.id,
        color="black" if creator_color is Color.BLACK else "white",
        token=token,
    )


@app.post("/api/games/{game_id}/join", response_model=PlayerCredentials)
async def join_game(
    game_id: str, body: JoinGameRequest, db: AsyncSession = Depends(get_session)
) -> PlayerCredentials:
    """Join a waiting human-vs-human game in the open seat."""
    session = manager.get(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    if session.mode != "hvh":
        raise HTTPException(400, "cannot join a bot game")
    if session.status != "waiting":
        raise HTTPException(409, "game already started")
    taken = set(session.tokens.values())
    open_color = Color.WHITE if Color.BLACK in taken else Color.BLACK
    token = session.add_player(open_color, body.user_id)
    session.status = "playing"
    row = await db.get(GameRow, game_id)
    if row is not None:
        row.status = "playing"
        if open_color is Color.BLACK:
            row.black_user_id = body.user_id
        else:
            row.white_user_id = body.user_id
        await db.commit()
    await session.broadcast_state()
    return PlayerCredentials(
        game_id=game_id,
        color="black" if open_color is Color.BLACK else "white",
        token=token,
    )


@app.get("/api/games/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: str) -> GameStateResponse:
    """Current state snapshot (works for restored games too)."""
    session = await manager.get_or_restore(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    return session.state()


@app.get("/api/games/{game_id}/history")
async def get_history(game_id: str) -> dict:
    """Board snapshots after every move, for history scrubbing in the UI.

    The engine replays the game server-side so the frontend never needs to
    implement capture logic.
    """
    session = await manager.get_or_restore(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    replay = Game(session.game.board_size, session.game.komi)
    boards = [board_rows(replay)]
    captures = [(0, 0)]
    for move in session.game.moves:
        if move.is_resign:
            break
        replay.play(move)
        boards.append(board_rows(replay))
        captures.append(
            (replay.captures[Color.BLACK], replay.captures[Color.WHITE])
        )
    return {
        "boards": boards,
        "captures": captures,
        "moves": [MovePayload.from_move(m).model_dump() for m in session.game.moves],
    }


@app.get("/api/games/{game_id}/sgf")
async def get_sgf(game_id: str) -> dict:
    """Export the game as SGF."""
    session = await manager.get_or_restore(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    return {"sgf": game_to_sgf(session.game)}


@app.get("/api/games")
async def list_games(
    status: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
) -> List[dict]:
    """Recent games, optionally filtered by status (waiting/playing/finished)."""
    query = select(GameRow).order_by(GameRow.created_at.desc()).limit(50)
    if status is not None:
        query = query.where(GameRow.status == status)
    rows = (await db.scalars(query)).all()
    return [
        {
            "game_id": r.id,
            "board_size": r.board_size,
            "mode": r.mode,
            "bot_level": r.bot_level,
            "status": r.status,
            "result": r.result,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.post("/api/games/{game_id}/analysis")
async def analyze_game(game_id: str) -> dict:
    """Win-rate + policy heatmap for the current position (via inference)."""
    session = await manager.get_or_restore(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    analysis = await request_analysis(session.game, session.bot_level)
    if analysis is None:
        raise HTTPException(503, "inference service unavailable")
    return analysis


@app.get("/api/bots/levels")
async def bot_levels() -> dict:
    """Available bot strength levels (proxied from the inference service)."""
    return {"levels": await list_bot_levels()}


# -- websocket -----------------------------------------------------------------


@app.websocket("/ws/games/{game_id}")
async def game_socket(
    websocket: WebSocket, game_id: str, token: Optional[str] = Query(default=None)
) -> None:
    """Real-time channel: receives move intents, broadcasts state snapshots.

    Clients without a valid token connect as spectators.
    """
    session = await manager.get_or_restore(game_id)
    if session is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    session.sockets.add(websocket)
    color = session.color_for_token(token)
    try:
        await websocket.send_json(
            {"type": "state", "state": session.state().model_dump()}
        )
        while True:
            data = await websocket.receive_json()
            kind = data.get("type")
            if kind == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if kind == "move":
                if color is None:
                    await websocket.send_json(
                        {"type": "error", "detail": "spectators cannot move"}
                    )
                    continue
                try:
                    payload = MovePayload(**data.get("move", {}))
                    await session.play_move(color, payload)
                except IllegalMoveError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                except (TypeError, ValueError) as exc:
                    await websocket.send_json(
                        {"type": "error", "detail": f"bad move payload: {exc}"}
                    )
            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"unknown message type: {kind}"}
                )
    except WebSocketDisconnect:
        pass
    finally:
        session.sockets.discard(websocket)
