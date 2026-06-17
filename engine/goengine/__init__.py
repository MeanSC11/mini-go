"""Pure-Python Go (Baduk) rules engine.

This package is the single source of truth for game rules. It is shared by
the training pipeline, the game server and the inference service.
"""

from goengine.types import Color, IllegalMoveError, Move, Point
from goengine.board import Board
from goengine.game import Game, GameResult
from goengine.eyes import candidate_moves, is_eye_fill
from goengine.sgf import game_to_sgf, sgf_to_game

__all__ = [
    "Color",
    "Move",
    "Point",
    "IllegalMoveError",
    "Board",
    "Game",
    "GameResult",
    "candidate_moves",
    "is_eye_fill",
    "game_to_sgf",
    "sgf_to_game",
]
