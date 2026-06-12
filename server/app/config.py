"""Server settings, read from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration. All fields can be overridden via env vars."""

    database_url: str = "postgresql+asyncpg://gobot:gobot@localhost:5432/gobot"
    inference_url: str = "http://localhost:8001"
    inference_timeout_seconds: float = 30.0
    cors_origins: list[str] = ["http://localhost:3000"]
    bot_default_level: str = "random"

    model_config = {"env_prefix": "GOBOT_"}


settings = Settings()
