"""Async SQLAlchemy models and session management."""

from __future__ import annotations

import datetime
import uuid
from typing import AsyncIterator, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    """Declarative base for all tables."""


def _uuid() -> str:
    return uuid.uuid4().hex


class UserRow(Base):
    """A registered player with an ELO rating."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    elo: Mapped[float] = mapped_column(Float, default=1200.0)
    games_played: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GameRow(Base):
    """A game record; ``sgf`` always reflects the latest known state."""

    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    board_size: Mapped[int] = mapped_column(Integer, default=9)
    komi: Mapped[float] = mapped_column(Float, default=7.5)
    mode: Mapped[str] = mapped_column(String(16))  # "hvh" | "hvb"
    bot_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="waiting")
    result: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    sgf: Mapped[str] = mapped_column(Text, default="")
    black_user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    white_user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


engine = create_async_engine(settings.database_url, echo=False)
session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Create tables if they do not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a database session."""
    async with session_factory() as session:
        yield session
