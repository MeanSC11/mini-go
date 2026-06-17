"""Dead-stone removal at scoring time (the training-blocker bug).

These are written first to prove the bug exists, then to lock in the fix.
Diagrams: X = black, O = white, . = empty. Black is given its own eyes at
(0,7) and (5,7) so it is unambiguously alive (a fully packed board would be
illegal -- the surrounding color would have no liberties).
"""

from goengine import Color, Game, Move

from conftest import build_game


def test_dead_ring_group_removed_one_eye() -> None:
    # A white "ring" group with exactly one eye at (2,2), fully surrounded by
    # black. White cannot make a second eye -> dead -> the board is black's.
    game = build_game(
        """
        X X X X X X X . X
        X O O O X X X X X
        X O . O X X X X X
        X O O O X X X X X
        X X X X X X X X X
        X X X X X X X . X
        X X X X X X X X X
        X X X X X X X X X
        X X X X X X X X X
        """,
        board_size=9,
        komi=7.5,
    )
    black, white = game.score()
    assert (black, white) == (81.0, 7.5)  # whole board to black, komi to white


def test_living_ring_group_two_eyes_kept() -> None:
    # Same shape but white has two eyes at (2,2) and (2,4): white is alive and
    # keeps its stones and territory (guard against over-removal).
    game = build_game(
        """
        X X X X X X X . X
        X O O O O O X X X
        X O . O . O X X X
        X O O O O O X X X
        X X X X X X X X X
        X X X X X X X . X
        X X X X X X X X X
        X X X X X X X X X
        X X X X X X X X X
        """,
        board_size=9,
        komi=7.5,
    )
    black, white = game.score()
    # White: 13 stones + 2 eyes = 15. Black: 81 - 15 = 66. White total + komi.
    assert (black, white) == (66.0, 22.5)


def test_dead_group_floating_with_liberties() -> None:
    # White stones floating in black's area, their liberties touching black
    # (the reported failure mode -- a group "obviously captured" that the old
    # scorer counted as alive). Dead -> removed -> board is black's.
    game = build_game(
        """
        X X X X X X X . X
        X X X X X X X X X
        X X X O O X X X X
        X X X . . X X X X
        X X X X X X X . X
        X X X X X X X X X
        X X X X X X X X X
        X X X X X X X X X
        X X X X X X X X X
        """,
        board_size=9,
        komi=7.5,
    )
    black, white = game.score()
    assert (black, white) == (81.0, 7.5)


def test_multiple_dead_white_groups_black_wins_clearly() -> None:
    # Mirrors the reported games: black dominates, several white groups are
    # surrounded and dead. The old scorer reported a near-tie / wrong winner by
    # counting the dead white stones (and their eye space) for white. With
    # dead-stone removal black must win by a wide margin.
    game = build_game(
        """
        X X X X X X X . X
        X O . X X . O X X
        X O X X X X X X X
        X X X X . O O . X
        X X X X X O . O X
        X . X X X O O O X
        X X X X X X X X X
        X X X X X X X . X
        X X X X X X X X X
        """,
        board_size=9,
        komi=7.5,
    )
    black, white = game.score()
    assert black > white  # winner is black, not a near-tie for white
    # Every white group is dead and removed; the board is entirely black.
    assert (black, white) == (81.0, 7.5)


def test_winner_at_game_end_uses_dead_stone_aware_scoring() -> None:
    # The end-of-game result (the label self-play feeds to training) must match
    # dead-stone-aware scoring. Two passes on a clearly-won-by-black board.
    game = build_game(
        """
        X X X X X X X . X
        X O O O X X X X X
        X O . O X X X X X
        X O O O X X X X X
        X X X X X X X . X
        X X X X X X X X X
        X X X X X X X X X
        X X X X X X X X X
        X X X X X X X X X
        """,
        board_size=9,
        komi=7.5,
    )
    game.play(Move.pass_turn())
    game.play(Move.pass_turn())
    assert game.result is not None
    assert game.result.winner is Color.BLACK
    assert game.result.margin == 81.0 - 7.5  # B+73.5, not a one-eye-group near-tie
