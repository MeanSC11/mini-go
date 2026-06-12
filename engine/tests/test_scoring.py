"""Chinese (area) scoring and end-of-game results."""

from goengine import Color, Game, Move

from conftest import build_game


def test_split_board_score() -> None:
    # Black wall on column 3, white wall on column 5; column 4 is dame.
    game = build_game(
        """
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        . . . X . O . . .
        """,
        komi=7.5,
    )
    black, white = game.score()
    # Black: 9 stones + 27 territory; White: 9 stones + 27 territory + komi.
    assert black == 36.0
    assert white == 43.5
    game.play(Move.pass_turn())
    game.play(Move.pass_turn())
    assert game.result is not None
    assert game.result.winner is Color.WHITE
    assert game.result.margin == 7.5
    assert str(game.result) == "W+7.5"


def test_black_wins_with_larger_area() -> None:
    # Black wall on column 6: black owns 54 points of territory + 9 stones.
    game = build_game(
        """
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        . . . . . . X O .
        """,
        komi=7.5,
    )
    black, white = game.score()
    assert black == 63.0  # 9 stones + 54 territory
    assert white == 9.0 + 9.0 + 7.5  # 9 stones + 9 territory (col 8) + komi
    game.play(Move.pass_turn())
    game.play(Move.pass_turn())
    assert game.result is not None
    assert game.result.winner is Color.BLACK
    assert game.result.margin == 63.0 - 25.5


def test_empty_board_is_all_dame() -> None:
    game = Game(9, komi=7.5)
    black, white = game.score()
    assert black == 0.0
    assert white == 7.5  # komi only; empty region touches no stones


def test_neutral_region_counts_for_nobody() -> None:
    # A single empty region touching both colors is dame.
    game = build_game(
        """
        X O . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . . . . .
        """,
        komi=0.0,
    )
    black, white = game.score()
    assert black == 1.0  # just the stone
    assert white == 1.0


def test_captured_area_counts_for_capturer() -> None:
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
        """,
        komi=0.0,
    )
    game.play(Move.play(0, 2))  # captures the two white stones
    black, white = game.score()
    # All empty points now border only black: whole board is black area.
    assert black == 81.0
    assert white == 0.0


def test_scoring_works_on_other_board_sizes() -> None:
    game = Game(13, komi=7.5)
    game.play(Move.play(6, 6))
    game.play(Move.pass_turn())
    game.play(Move.pass_turn())
    assert game.result is not None
    # Black owns the entire 13x13 board: 169 vs komi 7.5.
    assert game.result.winner is Color.BLACK
    assert game.result.margin == 169 - 7.5
