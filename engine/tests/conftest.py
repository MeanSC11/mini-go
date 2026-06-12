"""Shared test helpers."""

from __future__ import annotations

from typing import List

from goengine import Color, Game, Move


def build_game(diagram: str, board_size: int = 9, komi: float = 7.5) -> Game:
    """Build a game whose board matches ``diagram`` directly (bypassing play).

    ``diagram`` rows use ``X`` for black, ``O`` for white, ``.`` for empty.
    Useful for setting up scoring positions without legal move sequences.
    The position history is seeded with the resulting position.
    """
    game = Game(board_size=board_size, komi=komi)
    rows: List[str] = [line.split() for line in diagram.strip().splitlines()]  # type: ignore[misc]
    assert len(rows) == board_size, "diagram height mismatch"
    for r, row in enumerate(rows):
        assert len(row) == board_size, "diagram width mismatch"
        for c, cell in enumerate(row):
            if cell == "X":
                game.board._set((r, c), Color.BLACK)
            elif cell == "O":
                game.board._set((r, c), Color.WHITE)
    game._position_history = {game.board.position_hash}
    return game


def play_sequence(game: Game, points: List[tuple]) -> None:
    """Play alternating moves; ``None`` entries are passes."""
    for p in points:
        if p is None:
            game.play(Move.pass_turn())
        else:
            game.play(Move.play(*p))
