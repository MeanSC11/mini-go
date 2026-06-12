"""Inference HTTP API."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


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
