"""Pure-MCTS bot behavior."""

from goengine import Color, Game, Move

from app.uct import UctBot, is_simple_eye


def test_uct_returns_legal_move() -> None:
    game = Game(9)
    bot = UctBot(simulations=50, seed=1)
    move, win_rate, policy = bot.search(game)
    assert game.is_legal(move)
    assert 0.0 <= win_rate <= 1.0
    assert policy and abs(sum(policy.values()) - 1.0) < 1e-6


def test_uct_wins_capturing_race() -> None:
    # Capturing race: black's middle row is in atari at (1,4); the big white
    # group below is in atari at (4,0). Capturing at (4,0) is the only
    # winning move; anything else lets white capture five black stones.
    game = Game(5, komi=0.5)
    layout = {
        Color.WHITE: [(1, 0), (1, 1), (1, 2), (1, 3),
                      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                      (4, 1), (4, 2), (4, 3), (4, 4)],
        Color.BLACK: [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    }
    for color, points in layout.items():
        for point in points:
            game.board._set(point, color)
    game._position_history = {game.board.position_hash}
    bot = UctBot(simulations=400, seed=7)
    move, win_rate, _ = bot.search(game)
    assert move.point == (4, 0)
    assert win_rate > 0.5


def test_eye_detection() -> None:
    game = Game(9)
    # Black diamond around (1,1).
    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (0, 2), (2, 0), (2, 2)]:
        game.board._set((r, c), Color.BLACK)
    assert is_simple_eye(game, (1, 1), Color.BLACK)
    assert not is_simple_eye(game, (1, 1), Color.WHITE)
    assert not is_simple_eye(game, (5, 5), Color.BLACK)
