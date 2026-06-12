"""Rule enforcement: captures, suicide, ko (positional superko), snapback."""

import pytest

from goengine import Color, Game, IllegalMoveError, Move

from conftest import build_game


# -- captures ----------------------------------------------------------------


def test_single_stone_capture() -> None:
    game = Game(9)
    # Black surrounds the white stone at (4, 4).
    for move in [(4, 3), (4, 4), (3, 4), (8, 8), (5, 4), (8, 7), (4, 5)]:
        game.play(Move.play(*move))
    assert game.board.get((4, 4)) is Color.EMPTY
    assert game.captures[Color.BLACK] == 1
    assert game.captures[Color.WHITE] == 0


def test_capture_two_separate_groups_with_one_move() -> None:
    game = build_game(
        """
        . O X . . . . . .
        O X . . . . . . .
        X . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    # White stones at (0,1) and (1,0) are two distinct groups, both with
    # (0,0) as their only liberty.
    game.play(Move.play(0, 0))
    assert game.board.get((0, 1)) is Color.EMPTY
    assert game.board.get((1, 0)) is Color.EMPTY
    assert game.board.get((0, 0)) is Color.BLACK
    assert game.captures[Color.BLACK] == 2


def test_multi_stone_group_capture() -> None:
    # The two-stone white group has a single liberty at (0,2).
    game = build_game(
        """
        . X . X . . . . .
        X O O X . . . . .
        . X X . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    game.play(Move.play(0, 2))
    assert game.board.get((1, 1)) is Color.EMPTY
    assert game.board.get((1, 2)) is Color.EMPTY
    assert game.captures[Color.BLACK] == 2


# -- suicide -----------------------------------------------------------------


def test_single_stone_suicide_is_illegal() -> None:
    game = build_game(
        """
        . O . . . . . . .
        O . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    assert not game.is_legal(Move.play(0, 0))
    with pytest.raises(IllegalMoveError, match="suicide"):
        game.play(Move.play(0, 0))


def test_multi_stone_suicide_is_illegal() -> None:
    game = build_game(
        """
        X O . . . . . . .
        . O . . . . . . .
        O . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    # Black playing (1,0) would join (0,0) into a two-stone group with no
    # liberties and no captures: illegal.
    with pytest.raises(IllegalMoveError, match="suicide"):
        game.play(Move.play(1, 0))


def test_capturing_move_on_last_liberty_is_not_suicide() -> None:
    game = build_game(
        """
        . X O . . . . . .
        X O . O . . . . .
        . X O . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    # (1,2) is surrounded by white stones, but playing there captures (1,1).
    game.play(Move.play(1, 2))
    assert game.board.get((1, 1)) is Color.EMPTY
    assert game.board.get((1, 2)) is Color.BLACK
    assert game.captures[Color.BLACK] == 1


# -- ko / positional superko ---------------------------------------------------


def test_ko_immediate_recapture_is_illegal() -> None:
    game = build_game(
        """
        . X O . . . . . .
        X O . O . . . . .
        . X O . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    game.play(Move.play(1, 2))  # Black captures the ko stone at (1,1).
    assert game.board.get((1, 1)) is Color.EMPTY
    # White retaking at (1,1) would recreate the previous position.
    assert not game.is_legal(Move.play(1, 1))
    with pytest.raises(IllegalMoveError, match="superko"):
        game.play(Move.play(1, 1))


def test_ko_recapture_legal_after_threat_exchange() -> None:
    game = build_game(
        """
        . X O . . . . . .
        X O . O . . . . .
        . X O . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    game.play(Move.play(1, 2))  # B takes the ko.
    game.play(Move.play(5, 5))  # W plays a ko threat elsewhere.
    game.play(Move.play(5, 3))  # B answers.
    # The whole-board position after retaking now differs (extra stones),
    # so the recapture is legal under positional superko.
    game.play(Move.play(1, 1))  # W retakes the ko.
    assert game.board.get((1, 2)) is Color.EMPTY
    assert game.board.get((1, 1)) is Color.WHITE
    assert game.captures[Color.WHITE] == 1


# -- snapback ------------------------------------------------------------------


def test_snapback_recapture_is_legal() -> None:
    game = build_game(
        """
        . O X . . . . . .
        . O X . . . . . .
        X X . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """
    )
    game.play(Move.play(0, 0))  # B throw-in; the stone has one liberty (1,0).
    game.play(Move.play(1, 0))  # W captures the throw-in stone...
    assert game.board.get((0, 0)) is Color.EMPTY
    assert game.captures[Color.WHITE] == 1
    # ...but the capturing white group {(0,1),(1,1),(1,0)} now has a single
    # liberty at (0,0). Black recaptures three stones: not a ko, because the
    # resulting position is new.
    assert game.is_legal(Move.play(0, 0))
    game.play(Move.play(0, 0))
    assert game.captures[Color.BLACK] == 3
    for point in [(0, 1), (1, 1), (1, 0)]:
        assert game.board.get(point) is Color.EMPTY


# -- game flow -------------------------------------------------------------------


def test_two_passes_end_game() -> None:
    game = Game(9)
    game.play(Move.pass_turn())
    assert not game.is_over
    game.play(Move.pass_turn())
    assert game.is_over
    with pytest.raises(IllegalMoveError):
        game.play(Move.play(0, 0))


def test_move_resets_consecutive_passes() -> None:
    game = Game(9)
    game.play(Move.pass_turn())
    game.play(Move.play(4, 4))
    game.play(Move.pass_turn())
    assert not game.is_over


def test_resign_ends_game_immediately() -> None:
    game = Game(9)
    game.play(Move.play(4, 4))
    game.play(Move.resign())  # White resigns.
    assert game.is_over
    assert game.result is not None
    assert game.result.winner is Color.BLACK
    assert game.result.by_resignation
    assert str(game.result) == "B+R"


def test_legal_moves_on_empty_board() -> None:
    game = Game(9)
    moves = game.legal_moves()
    assert len(moves) == 82  # 81 points + pass
    assert any(m.is_pass for m in moves)


def test_copy_is_independent() -> None:
    game = Game(9)
    game.play(Move.play(4, 4))
    clone = game.copy()
    clone.play(Move.play(3, 3))
    assert game.board.get((3, 3)) is Color.EMPTY
    assert clone.board.get((3, 3)) is Color.WHITE
    assert len(game.moves) == 1
    assert len(clone.moves) == 2
