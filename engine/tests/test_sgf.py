"""SGF export / import round-trips."""

import pytest

from goengine import Color, Game, Move, game_to_sgf, sgf_to_game


def _play_short_game() -> Game:
    game = Game(9, komi=7.5)
    for move in [(2, 2), (6, 6), (2, 6), (6, 2), (4, 4)]:
        game.play(Move.play(*move))
    game.play(Move.pass_turn())
    game.play(Move.pass_turn())
    return game


def test_export_contains_metadata_and_moves() -> None:
    sgf = game_to_sgf(_play_short_game())
    assert sgf.startswith("(;")
    assert "SZ[9]" in sgf
    assert "KM[7.5]" in sgf
    assert "RU[Chinese]" in sgf
    assert ";B[cc]" in sgf  # (2,2) -> col c, row c
    assert ";W[gg]" in sgf
    assert ";W[]" in sgf  # pass
    assert "RE[" in sgf


def test_roundtrip_preserves_position_and_result() -> None:
    original = _play_short_game()
    restored = sgf_to_game(game_to_sgf(original))
    assert restored.board_size == original.board_size
    assert restored.komi == original.komi
    assert restored.board.position_hash == original.board.position_hash
    assert restored.is_over == original.is_over
    assert restored.result is not None and original.result is not None
    assert str(restored.result) == str(original.result)


def test_roundtrip_with_captures() -> None:
    game = Game(9)
    for move in [(4, 3), (4, 4), (3, 4), (8, 8), (5, 4), (8, 7), (4, 5)]:
        game.play(Move.play(*move))
    restored = sgf_to_game(game_to_sgf(game))
    assert restored.board.get((4, 4)) is Color.EMPTY
    assert restored.captures[Color.BLACK] == 1
    assert restored.board.position_hash == game.board.position_hash


def test_roundtrip_resignation() -> None:
    game = Game(9)
    game.play(Move.play(4, 4))
    game.play(Move.resign())
    restored = sgf_to_game(game_to_sgf(game))
    assert restored.is_over
    assert restored.result is not None
    assert restored.result.by_resignation
    assert restored.result.winner is Color.BLACK


def test_import_supports_13x13_and_19x19() -> None:
    for size in (13, 19):
        game = Game(size, komi=7.5)
        game.play(Move.play(3, 3))
        restored = sgf_to_game(game_to_sgf(game))
        assert restored.board_size == size
        assert restored.board.get((3, 3)) is Color.BLACK


def test_import_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        sgf_to_game("this is not sgf")


def test_import_rejects_out_of_turn_moves() -> None:
    with pytest.raises(ValueError, match="out-of-turn"):
        sgf_to_game("(;GM[1]SZ[9];W[aa])")


def test_import_accepts_tt_as_pass_on_small_boards() -> None:
    game = sgf_to_game("(;GM[1]SZ[9]KM[7.5];B[cc];W[tt];B[tt])")
    # B move + two passes -> game over.
    assert game.is_over
