"""Minimal SGF (Smart Game Format) export/import for engine games.

Supports the properties the project needs: board size (SZ), komi (KM),
result (RE) and the move sequence (B/W). Pass moves are encoded as empty
values (``B[]``), the modern convention. ``tt`` is also accepted as a pass
on import for boards up to 19x19, for compatibility with older files.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from goengine.game import Game
from goengine.types import Color, Move

_SGF_COLS = "abcdefghijklmnopqrstuvwxy"


def _point_to_sgf(point: Tuple[int, int]) -> str:
    row, col = point
    return f"{_SGF_COLS[col]}{_SGF_COLS[row]}"


def _sgf_to_point(value: str) -> Tuple[int, int]:
    col = _SGF_COLS.index(value[0])
    row = _SGF_COLS.index(value[1])
    return row, col


def game_to_sgf(game: Game) -> str:
    """Serialize ``game`` (root properties + move sequence) to an SGF string."""
    parts: List[str] = [
        "(;GM[1]FF[4]CA[UTF-8]AP[go-bot]",
        f"SZ[{game.board_size}]",
        f"KM[{game.komi:g}]",
        "RU[Chinese]",
    ]
    if game.result is not None:
        parts.append(f"RE[{game.result}]")
    color = Color.BLACK
    for move in game.moves:
        if move.is_resign:
            break
        tag = "B" if color is Color.BLACK else "W"
        value = "" if move.is_pass else _point_to_sgf(move.point)  # type: ignore[arg-type]
        parts.append(f";{tag}[{value}]")
        color = color.opponent
    parts.append(")")
    return "".join(parts)


_PROP_RE = re.compile(r"([A-Z]{1,2})((?:\[[^\]]*\])+)")
_VALUE_RE = re.compile(r"\[([^\]]*)\]")


def sgf_to_game(sgf: str) -> Game:
    """Parse an SGF string and replay it into a :class:`Game`.

    Raises ``ValueError`` on malformed input and ``IllegalMoveError`` if the
    recorded moves are illegal under this engine's rules.
    """
    if "(" not in sgf or ")" not in sgf:
        raise ValueError("not an SGF game tree")
    size = 19
    komi = 0.0
    moves: List[Tuple[str, str]] = []
    resigned = False
    for match in _PROP_RE.finditer(sgf):
        ident = match.group(1)
        values = _VALUE_RE.findall(match.group(2))
        if ident == "SZ":
            size = int(values[0])
        elif ident == "KM":
            komi = float(values[0])
        elif ident == "RE" and values[0].endswith("+R"):
            resigned = True
        elif ident in ("B", "W"):
            moves.append((ident, values[0]))
    game = Game(board_size=size, komi=komi)
    for tag, value in moves:
        expected = "B" if game.current_player is Color.BLACK else "W"
        if tag != expected:
            raise ValueError(f"out-of-turn move {tag}[{value}]")
        if value == "" or (value == "tt" and size <= 19):
            game.play(Move.pass_turn())
        else:
            row, col = _sgf_to_point(value)
            game.play(Move.play(row, col))
    if resigned and not game.is_over:
        game.play(Move.resign())
    return game
