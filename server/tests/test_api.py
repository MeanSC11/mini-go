"""End-to-end API tests against a SQLite-backed app instance.

Production uses PostgreSQL (see docker-compose); SQLite keeps tests
dependency-free. The SQL layer is identical SQLAlchemy code.
"""

from __future__ import annotations

import os

os.environ["GOBOT_DATABASE_URL"] = "sqlite+aiosqlite://"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _create_hvh_game(client: TestClient) -> tuple[str, str, str]:
    created = client.post("/api/games", json={"board_size": 9, "mode": "hvh"})
    assert created.status_code == 200
    black = created.json()
    joined = client.post(f"/api/games/{black['game_id']}/join", json={})
    assert joined.status_code == 200
    white = joined.json()
    assert {black["color"], white["color"]} == {"black", "white"}
    return black["game_id"], black["token"], white["token"]


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_join_and_state(client: TestClient) -> None:
    game_id, _, _ = _create_hvh_game(client)
    state = client.get(f"/api/games/{game_id}").json()
    assert state["status"] == "playing"
    assert state["board_size"] == 9
    assert state["current_player"] == "black"
    assert len(state["board"]) == 9


def test_websocket_play_and_finish(client: TestClient) -> None:
    game_id, black_token, white_token = _create_hvh_game(client)
    with client.websocket_connect(
        f"/ws/games/{game_id}?token={black_token}"
    ) as black_ws, client.websocket_connect(
        f"/ws/games/{game_id}?token={white_token}"
    ) as white_ws:
        assert black_ws.receive_json()["type"] == "state"
        assert white_ws.receive_json()["type"] == "state"

        # Black plays (2,2); both sockets get the new state.
        black_ws.send_json(
            {"type": "move", "move": {"type": "play", "row": 2, "col": 2}}
        )
        msg = black_ws.receive_json()
        assert msg["type"] == "state"
        assert msg["state"]["board"][2][2] == "B"
        assert white_ws.receive_json()["state"]["current_player"] == "white"

        # White cannot play on the occupied point.
        white_ws.send_json(
            {"type": "move", "move": {"type": "play", "row": 2, "col": 2}}
        )
        err = white_ws.receive_json()
        assert err["type"] == "error"

        # Out-of-turn move by black is rejected.
        black_ws.send_json(
            {"type": "move", "move": {"type": "play", "row": 3, "col": 3}}
        )
        assert black_ws.receive_json()["type"] == "error"

        # Two passes end the game.
        white_ws.send_json({"type": "move", "move": {"type": "pass"}})
        assert black_ws.receive_json()["type"] == "state"
        white_ws.receive_json()
        black_ws.send_json({"type": "move", "move": {"type": "pass"}})
        final = black_ws.receive_json()["state"]
        assert final["status"] == "finished"
        assert final["result"] is not None
        assert final["score_black"] is not None

    # Game is persisted with SGF.
    sgf = client.get(f"/api/games/{game_id}/sgf").json()["sgf"]
    assert "SZ[9]" in sgf and ";B[cc]" in sgf
    games = client.get("/api/games", params={"status": "finished"}).json()
    assert any(g["game_id"] == game_id for g in games)


def test_history_endpoint(client: TestClient) -> None:
    game_id, black_token, white_token = _create_hvh_game(client)
    with client.websocket_connect(
        f"/ws/games/{game_id}?token={black_token}"
    ) as black_ws, client.websocket_connect(
        f"/ws/games/{game_id}?token={white_token}"
    ) as white_ws:
        black_ws.receive_json()
        white_ws.receive_json()
        black_ws.send_json(
            {"type": "move", "move": {"type": "play", "row": 0, "col": 0}}
        )
        black_ws.receive_json()
        white_ws.receive_json()
    history = client.get(f"/api/games/{game_id}/history").json()
    assert len(history["boards"]) == 2  # initial position + 1 move
    assert history["boards"][0][0][0] == "."
    assert history["boards"][1][0][0] == "B"


def test_resign_ends_game(client: TestClient) -> None:
    game_id, black_token, white_token = _create_hvh_game(client)
    with client.websocket_connect(
        f"/ws/games/{game_id}?token={white_token}"
    ) as white_ws:
        white_ws.receive_json()
        white_ws.send_json({"type": "move", "move": {"type": "resign"}})
        final = white_ws.receive_json()["state"]
        assert final["status"] == "finished"
        assert final["result"] == "B+R"


def test_bot_game_with_random_bot(client: TestClient) -> None:
    created = client.post(
        "/api/games", json={"board_size": 9, "mode": "hvb", "bot_level": "random"}
    ).json()
    game_id, token = created["game_id"], created["token"]
    with client.websocket_connect(f"/ws/games/{game_id}?token={token}") as ws:
        assert ws.receive_json()["state"]["status"] == "playing"
        ws.send_json({"type": "move", "move": {"type": "play", "row": 4, "col": 4}})
        assert ws.receive_json()["state"]["board"][4][4] == "B"
        # The random bot answers asynchronously; next broadcast has its move.
        after_bot = ws.receive_json()["state"]
        assert after_bot["current_player"] == "black"
        assert sum(row.count("W") for row in after_bot["board"]) == 1


def test_spectator_cannot_move(client: TestClient) -> None:
    game_id, _, _ = _create_hvh_game(client)
    with client.websocket_connect(f"/ws/games/{game_id}") as spectator:
        spectator.receive_json()
        spectator.send_json(
            {"type": "move", "move": {"type": "play", "row": 0, "col": 0}}
        )
        assert spectator.receive_json()["type"] == "error"


def test_users_and_leaderboard(client: TestClient) -> None:
    user = client.post("/api/users", json={"name": "alice"}).json()
    assert user["elo"] == 1200.0
    again = client.post("/api/users", json={"name": "alice"}).json()
    assert again["id"] == user["id"]
    board = client.get("/api/users").json()
    assert any(u["name"] == "alice" for u in board)
