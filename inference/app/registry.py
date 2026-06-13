"""Bot registry: maps level names to bot instances.

Levels:
  * ``random``            – uniform random legal move
  * ``mcts-<N>``          – pure UCT MCTS with N simulations per move
  * ``az-<checkpoint>``   – AlphaZero network + PUCT MCTS, one level per
                            checkpoint file found in ``checkpoint_dir``
                            (older checkpoints are weaker)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from pydantic_settings import BaseSettings

from goengine import Game, Move

from app.uct import UctBot

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Inference service configuration."""

    checkpoint_dir: str = "checkpoints"
    az_simulations: int = 160
    az_time_per_move: Optional[float] = None  # seconds/move; overrides sims when set
    mcts_levels: List[int] = [100, 300, 800]
    device: str = "cpu"

    model_config = {"env_prefix": "GOBOT_INF_"}


settings = Settings()


class Bot(Protocol):
    """A bot returns (move, side-to-move win rate or None, policy)."""

    def search(self, game: Game) -> Tuple[Move, Optional[float], Dict[str, float]]:
        ...


class RandomBot:
    """Uniform random legal move; the Phase-2 placeholder level."""

    def search(self, game: Game) -> Tuple[Move, Optional[float], Dict[str, float]]:
        """Pick a random legal non-pass move (pass only when forced)."""
        moves = game.legal_moves(include_pass=False)
        if not moves:
            return Move.pass_turn(), None, {}
        return random.choice(moves), None, {}


class _UctAdapter:
    """Adapts UctBot to the registry protocol."""

    def __init__(self, simulations: int) -> None:
        self._bot = UctBot(simulations=simulations)

    def search(self, game: Game) -> Tuple[Move, Optional[float], Dict[str, float]]:
        move, win_rate, policy = self._bot.search(game)
        return move, win_rate, policy


class _AzeroAdapter:
    """Lazy-loading AlphaZero bot bound to one checkpoint file."""

    def __init__(
        self, checkpoint_path: Path, simulations: int, device: str,
        time_limit: Optional[float] = None,
    ) -> None:
        self._path = checkpoint_path
        self._simulations = simulations
        self._device = device
        self._time_limit = time_limit
        self._player = None

    def _load(self):
        if self._player is None:
            from azero.bot import AlphaZeroPlayer  # requires torch

            self._player = AlphaZeroPlayer(
                str(self._path), simulations=self._simulations, device=self._device,
                time_limit=self._time_limit,
            )
        return self._player

    def search(self, game: Game) -> Tuple[Move, Optional[float], Dict[str, float]]:
        return self._load().search(game)


def discover_checkpoints() -> List[Path]:
    """Checkpoint files sorted by name (training names them by iteration)."""
    directory = Path(settings.checkpoint_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.pt"))


def available_levels() -> List[dict]:
    """Describe every playable level."""
    levels = [{"name": "random", "description": "Random legal moves"}]
    for sims in settings.mcts_levels:
        levels.append(
            {
                "name": f"mcts-{sims}",
                "description": f"Pure MCTS, {sims} simulations/move",
            }
        )
    for path in discover_checkpoints():
        levels.append(
            {
                "name": f"az-{path.stem}",
                "description": f"AlphaZero network ({path.stem})",
            }
        )
    return levels


_cache: Dict[str, Bot] = {}


def get_bot(level: str) -> Bot:
    """Resolve a level name to a bot instance; unknown levels -> random."""
    bot = _cache.get(level)
    if bot is not None:
        return bot
    if level.startswith("mcts-"):
        try:
            sims = int(level.split("-", 1)[1])
        except ValueError:
            sims = 300
        bot = _UctAdapter(simulations=sims)
    elif level.startswith("az-"):
        stem = level[3:]
        path = Path(settings.checkpoint_dir) / f"{stem}.pt"
        if path.is_file():
            bot = _AzeroAdapter(
                path, settings.az_simulations, settings.device,
                time_limit=settings.az_time_per_move,
            )
        else:
            logger.warning("checkpoint %s not found; using random bot", path)
            bot = RandomBot()
    else:
        bot = RandomBot()
    _cache[level] = bot
    return bot
