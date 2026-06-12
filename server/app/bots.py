"""Bot move providers.

The real bot lives in the inference service (Phase 3+); the server talks to
it over HTTP. If the service is unreachable, or the game was created with
``bot_level == "random"``, a local random bot is used as a fallback so
human-vs-bot games always work.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from goengine import Game, Move

from app.config import settings
from app.schemas import MovePayload

logger = logging.getLogger(__name__)


@dataclass
class BotReply:
    """A bot's chosen move plus optional analysis for the UI."""

    move: Move
    win_rate: Optional[float] = None
    policy: Dict[str, float] = field(default_factory=dict)


def random_move(game: Game) -> Move:
    """Pick a uniformly random legal move, preferring not to pass."""
    moves = [m for m in game.legal_moves(include_pass=False)]
    if not moves:
        return Move.pass_turn()
    return random.choice(moves)


async def request_bot_move(game: Game, bot_level: Optional[str]) -> BotReply:
    """Ask the inference service for a move; fall back to the random bot."""
    if bot_level in (None, "random"):
        return BotReply(move=random_move(game))
    payload = {
        "board_size": game.board_size,
        "komi": game.komi,
        "moves": [MovePayload.from_move(m).model_dump() for m in game.moves],
        "level": bot_level,
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.inference_timeout_seconds
        ) as client:
            response = await client.post(
                f"{settings.inference_url}/move", json=payload
            )
            response.raise_for_status()
            data = response.json()
        move = MovePayload(**data["move"]).to_move()
        if not game.is_legal(move):
            logger.warning("inference returned illegal move %s; using random", move)
            return BotReply(move=random_move(game))
        return BotReply(
            move=move,
            win_rate=data.get("win_rate"),
            policy=data.get("policy") or {},
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("inference service unavailable (%s); using random bot", exc)
        return BotReply(move=random_move(game))


async def request_analysis(game: Game, bot_level: Optional[str]) -> Optional[dict]:
    """Ask the inference service to analyze the current position.

    Returns ``{"win_rate": float, "policy": {"r,c": prob}}`` or None if the
    service is unavailable.
    """
    payload = {
        "board_size": game.board_size,
        "komi": game.komi,
        "moves": [MovePayload.from_move(m).model_dump() for m in game.moves],
        "level": bot_level or settings.bot_default_level,
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.inference_timeout_seconds
        ) as client:
            response = await client.post(
                f"{settings.inference_url}/analyze", json=payload
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.warning("analysis unavailable: %s", exc)
        return None


async def list_bot_levels() -> List[dict]:
    """Fetch available bot levels from the inference service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.inference_url}/levels")
            response.raise_for_status()
            return response.json()["levels"]
    except httpx.HTTPError:
        return [{"name": "random", "description": "Random legal moves"}]
