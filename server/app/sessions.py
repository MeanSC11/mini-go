"""In-memory game sessions: engine state, connected clients, bot turns.

The engine is the single source of truth for rules. Clients only send move
intents; every move is validated here before being applied and broadcast.
Each applied move is persisted to the database (status, result and SGF).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from typing import Dict, List, Optional, Set

from fastapi import WebSocket

from goengine import Color, Game, IllegalMoveError, Move, game_to_sgf, sgf_to_game
from goengine.game import GameResult

from app import elo
from app.bots import BotReply, request_bot_move
from app.db import GameRow, UserRow, session_factory
from app.schemas import GameStateResponse, MovePayload, board_rows, color_name

logger = logging.getLogger(__name__)

# Nominal ratings for bot opponents, used to update human ELO in hvb games.
BOT_RATINGS: Dict[str, float] = {"random": 800.0}
DEFAULT_BOT_RATING = 1200.0


class GameSession:
    """One active game: engine state plus connection/auth bookkeeping."""

    def __init__(
        self,
        game_id: str,
        game: Game,
        mode: str,
        bot_level: Optional[str] = None,
        bot_color: Color = Color.WHITE,
    ) -> None:
        self.id = game_id
        self.game = game
        self.mode = mode
        self.bot_level = bot_level
        self.bot_color = bot_color
        self.status = "waiting"
        self.tokens: Dict[str, Color] = {}
        self.user_ids: Dict[Color, Optional[str]] = {
            Color.BLACK: None,
            Color.WHITE: None,
        }
        self.sockets: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.last_analysis: Optional[dict] = None

    # -- auth ----------------------------------------------------------------

    def add_player(self, color: Color, user_id: Optional[str]) -> str:
        """Register a player seat and return their move-authorizing token."""
        token = secrets.token_urlsafe(16)
        self.tokens[token] = color
        self.user_ids[color] = user_id
        return token

    def color_for_token(self, token: Optional[str]) -> Optional[Color]:
        """Resolve a token to a seat color (None for spectators)."""
        if token is None:
            return None
        return self.tokens.get(token)

    @property
    def seats_filled(self) -> bool:
        """Whether both colors have a controller (human or bot)."""
        humans = set(self.tokens.values())
        if self.mode == "hvb":
            humans.add(self.bot_color)
        return Color.BLACK in humans and Color.WHITE in humans

    # -- state ---------------------------------------------------------------

    def state(self) -> GameStateResponse:
        """Snapshot of the current game state."""
        score_black: Optional[float] = None
        score_white: Optional[float] = None
        if self.game.is_over and self.game.result is not None:
            if not self.game.result.by_resignation:
                score_black, score_white = self.game.score()
        return GameStateResponse(
            game_id=self.id,
            board_size=self.game.board_size,
            komi=self.game.komi,
            mode=self.mode,
            bot_level=self.bot_level,
            status=self.status,
            board=board_rows(self.game),
            current_player=color_name(self.game.current_player),
            captures_black=self.game.captures[Color.BLACK],
            captures_white=self.game.captures[Color.WHITE],
            moves=[MovePayload.from_move(m) for m in self.game.moves],
            result=str(self.game.result) if self.game.result else None,
            score_black=score_black,
            score_white=score_white,
        )

    # -- broadcasting ----------------------------------------------------------

    async def broadcast(self, message: dict) -> None:
        """Send ``message`` to every connected websocket, dropping dead ones."""
        dead: List[WebSocket] = []
        for ws in self.sockets:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - any send failure drops the socket
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    async def broadcast_state(self) -> None:
        """Broadcast the full state snapshot."""
        await self.broadcast({"type": "state", "state": self.state().model_dump()})

    # -- moves -------------------------------------------------------------------

    async def play_move(self, color: Color, payload: MovePayload) -> None:
        """Validate and apply a move for ``color``; raise IllegalMoveError if bad."""
        async with self.lock:
            if self.status == "waiting":
                raise IllegalMoveError("waiting for an opponent to join")
            if self.game.is_over:
                raise IllegalMoveError("game is over")
            move = payload.to_move()
            if move.is_resign:
                # Resigning is allowed out of turn, as the token's color.
                self._resign_as(color)
            else:
                if self.game.current_player is not color:
                    raise IllegalMoveError("not your turn")
                self.game.play(move)
            if self.game.is_over:
                self.status = "finished"
        await self._persist()
        await self.broadcast_state()
        if self.game.is_over:
            await self._finalize_ratings()
        elif self._is_bot_turn():
            asyncio.create_task(self._play_bot_turn())

    def _resign_as(self, color: Color) -> None:
        """Resign on behalf of ``color`` regardless of whose turn it is."""
        self.game.moves.append(Move.resign())
        self.game.result = GameResult(
            winner=color.opponent, margin=None, by_resignation=True
        )

    def maybe_start_bot_turn(self) -> None:
        """Schedule the bot's move if it is the bot's turn."""
        if self._is_bot_turn():
            asyncio.create_task(self._play_bot_turn())

    def _is_bot_turn(self) -> bool:
        return (
            self.mode == "hvb"
            and not self.game.is_over
            and self.game.current_player is self.bot_color
        )

    async def _play_bot_turn(self) -> None:
        """Compute and apply the bot's move, then broadcast."""
        try:
            reply: BotReply = await request_bot_move(self.game, self.bot_level)
            async with self.lock:
                if self.game.is_over or self.game.current_player is not self.bot_color:
                    return
                self.game.play(reply.move)
                if self.game.is_over:
                    self.status = "finished"
                if reply.win_rate is not None or reply.policy:
                    self.last_analysis = {
                        "win_rate": reply.win_rate,
                        "policy": reply.policy,
                    }
            await self._persist()
            await self.broadcast_state()
            if self.last_analysis:
                await self.broadcast(
                    {"type": "analysis", "analysis": self.last_analysis}
                )
            if self.game.is_over:
                await self._finalize_ratings()
        except Exception:  # noqa: BLE001
            logger.exception("bot turn failed for game %s", self.id)

    # -- persistence -----------------------------------------------------------

    async def _persist(self) -> None:
        """Write current status/result/SGF to the database."""
        async with session_factory() as session:
            row = await session.get(GameRow, self.id)
            if row is None:
                return
            row.status = self.status
            row.result = str(self.game.result) if self.game.result else None
            row.sgf = game_to_sgf(self.game)
            await session.commit()

    async def _finalize_ratings(self) -> None:
        """Update ELO ratings after a finished game."""
        result = self.game.result
        if result is None or result.winner is Color.EMPTY:
            return
        black_id = self.user_ids[Color.BLACK]
        white_id = self.user_ids[Color.WHITE]
        async with session_factory() as session:
            if self.mode == "hvh" and black_id and white_id:
                black = await session.get(UserRow, black_id)
                white = await session.get(UserRow, white_id)
                if black is None or white is None:
                    return
                score = 1.0 if result.winner is Color.BLACK else 0.0
                black.elo, white.elo = elo.update(black.elo, white.elo, score)
                black.games_played += 1
                white.games_played += 1
            elif self.mode == "hvb":
                human_color = (
                    Color.BLACK if self.bot_color is Color.WHITE else Color.WHITE
                )
                human_id = self.user_ids[human_color]
                if not human_id:
                    return
                human = await session.get(UserRow, human_id)
                if human is None:
                    return
                bot_rating = BOT_RATINGS.get(
                    self.bot_level or "random", DEFAULT_BOT_RATING
                )
                score = 1.0 if result.winner is human_color else 0.0
                human.elo, _ = elo.update(human.elo, bot_rating, score)
                human.games_played += 1
            else:
                return
            await session.commit()


class SessionManager:
    """Registry of active sessions; lazily restores finished/idle games from DB."""

    def __init__(self) -> None:
        self._sessions: Dict[str, GameSession] = {}

    def create(
        self,
        board_size: int,
        komi: float,
        mode: str,
        bot_level: Optional[str],
        bot_color: Color = Color.WHITE,
    ) -> GameSession:
        """Create a new in-memory session (caller persists the GameRow)."""
        game_id = uuid.uuid4().hex
        session = GameSession(
            game_id=game_id,
            game=Game(board_size=board_size, komi=komi),
            mode=mode,
            bot_level=bot_level,
            bot_color=bot_color,
        )
        self._sessions[game_id] = session
        return session

    def get(self, game_id: str) -> Optional[GameSession]:
        """Look up an active session."""
        return self._sessions.get(game_id)

    async def get_or_restore(self, game_id: str) -> Optional[GameSession]:
        """Get a session, restoring it from the database SGF if needed.

        Restored sessions have no player tokens (server restarts invalidate
        seats); they are usable for spectating and history review.
        """
        session = self._sessions.get(game_id)
        if session is not None:
            return session
        async with session_factory() as db:
            row = await db.get(GameRow, game_id)
        if row is None:
            return None
        game = sgf_to_game(row.sgf) if row.sgf else Game(row.board_size, row.komi)
        restored = GameSession(
            game_id=game_id,
            game=game,
            mode=row.mode,
            bot_level=row.bot_level,
        )
        restored.status = row.status
        self._sessions[game_id] = restored
        return restored


manager = SessionManager()
