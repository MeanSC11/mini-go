"""Inference HTTP API."""

from fastapi.testclient import TestClient

from goengine import Game, Move

from app.main import _should_resign, app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_should_resign_only_when_lost_past_opening() -> None:
    game = Game(9)
    # Opening: never resign, even with a low estimate.
    assert _should_resign(game, 0.01) is False
    # Past the opening (>= 2*board_size moves) and clearly lost -> resign.
    for _ in range(2 * game.board_size):
        game.play(Move.pass_turn() if game.is_over else _first_legal(game))
    assert _should_resign(game, 0.02) is True
    assert _should_resign(game, 0.5) is False  # a playable game is not resigned
    assert _should_resign(game, None) is False  # no estimate (random bot)


def _first_legal(game: Game) -> Move:
    for r in range(game.board_size):
        for c in range(game.board_size):
            move = Move.play(r, c)
            if game.is_legal(move):
                return move
    return Move.pass_turn()


def test_levels_include_random_and_mcts() -> None:
    names = [lv["name"] for lv in client.get("/levels").json()["levels"]]
    assert "random" in names
    assert any(name.startswith("mcts-") for name in names)


def test_random_move() -> None:
    response = client.post(
        "/move", json={"board_size": 9, "komi": 7.5, "moves": [], "level": "random"}
    )
    assert response.status_code == 200
    move = response.json()["move"]
    assert move["type"] == "play"
    assert 0 <= move["row"] < 9


def test_mcts_move_with_history() -> None:
    response = client.post(
        "/move",
        json={
            "board_size": 9,
            "komi": 7.5,
            "moves": [{"type": "play", "row": 4, "col": 4}],
            "level": "mcts-30",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["move"]["type"] in ("play", "pass")
    assert body["win_rate"] is not None
    assert body["policy"]


def test_invalid_history_rejected() -> None:
    response = client.post(
        "/move",
        json={
            "board_size": 9,
            "moves": [
                {"type": "play", "row": 0, "col": 0},
                {"type": "play", "row": 0, "col": 0},
            ],
            "level": "random",
        },
    )
    assert response.status_code == 422


def test_finished_game_rejected() -> None:
    response = client.post(
        "/move",
        json={
            "board_size": 9,
            "moves": [{"type": "pass"}, {"type": "pass"}],
            "level": "random",
        },
    )
    assert response.status_code == 422


def test_analyze_returns_policy() -> None:
    response = client.post(
        "/analyze",
        json={"board_size": 9, "moves": [], "level": "mcts-20"},
    )
    assert response.status_code == 200
    assert response.json()["policy"]
