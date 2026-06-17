"""Eye detection and pass-when-finished behavior."""

from goengine import Color, Game, Move, candidate_moves, is_eye_fill


def _fill(game: Game, color: Color, points) -> None:
    for p in points:
        game.board._set(p, color)
    game._position_history = {game.board.position_hash}


def test_interior_true_eye() -> None:
    game = Game(5)
    # Black surrounds (2,2) orthogonally and on all diagonals.
    _fill(game, Color.BLACK,
          [(1, 2), (3, 2), (2, 1), (2, 3), (1, 1), (1, 3), (3, 1), (3, 3)])
    assert is_eye_fill(game.board, (2, 2), Color.BLACK)
    assert not is_eye_fill(game.board, (2, 2), Color.WHITE)


def test_interior_false_eye_two_enemy_diagonals() -> None:
    game = Game(5)
    # Orthogonals black, but two diagonals are white -> false eye (playable).
    _fill(game, Color.BLACK, [(1, 2), (3, 2), (2, 1), (2, 3), (1, 1), (3, 3)])
    _fill(game, Color.WHITE, [(1, 3), (3, 1)])
    assert not is_eye_fill(game.board, (2, 2), Color.BLACK)


def test_corner_eye_needs_all_diagonals() -> None:
    game = Game(5)
    # Corner (0,0): orthogonals (0,1),(1,0) black, diagonal (1,1) black -> eye.
    _fill(game, Color.BLACK, [(0, 1), (1, 0), (1, 1)])
    assert is_eye_fill(game.board, (0, 0), Color.BLACK)
    # One enemy on the single corner diagonal -> false eye.
    game2 = Game(5)
    _fill(game2, Color.BLACK, [(0, 1), (1, 0)])
    game2.board._set((1, 1), Color.WHITE)
    game2._position_history = {game2.board.position_hash}
    assert not is_eye_fill(game2.board, (0, 0), Color.BLACK)


def test_open_point_is_not_an_eye() -> None:
    game = Game(5)
    _fill(game, Color.BLACK, [(2, 1)])  # only one neighbor filled
    assert not is_eye_fill(game.board, (2, 2), Color.BLACK)


def test_candidate_moves_only_pass_when_dominated() -> None:
    """A fully-controlled side with only its own eyes left has pass as its
    only candidate — so the bot passes instead of filling the board."""
    game = Game(5)
    black = [(r, c) for r in range(5) for c in range(5) if (r, c) not in {(1, 1), (3, 3)}]
    _fill(game, Color.BLACK, black)  # all black except two true eyes
    game.current_player = Color.BLACK
    # both empty points are genuine eyes...
    assert is_eye_fill(game.board, (1, 1), Color.BLACK)
    assert is_eye_fill(game.board, (3, 3), Color.BLACK)
    # ...and they are otherwise legal, so legal_moves would still offer them.
    assert any(not m.is_pass for m in game.legal_moves())
    # candidate_moves drops them, leaving only pass.
    cands = candidate_moves(game)
    assert cands == [Move.pass_turn()]


def test_two_passes_end_game_and_score() -> None:
    game = Game(5)
    # Black controls the board; a couple of white stones present but dominated.
    black = [(r, c) for r in range(5) for c in range(3)]  # left three columns
    _fill(game, Color.BLACK, black)
    game.current_player = Color.BLACK
    assert not game.is_over
    game.play(Move.pass_turn())
    game.play(Move.pass_turn())
    assert game.is_over
    assert game.result is not None
    black_pts, white_pts = game.score()
    expected = Color.BLACK if black_pts > white_pts else Color.WHITE
    assert game.result.winner is expected
