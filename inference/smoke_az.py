"""Manual smoke check: inference service serving an AlphaZero checkpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
levels = [lv["name"] for lv in client.get("/levels").json()["levels"]]
print("levels:", levels)
assert any(name.startswith("az-") for name in levels), "no az- levels found"

resp = client.post(
    "/move",
    json={
        "board_size": 5,
        "komi": 7.5,
        "moves": [{"type": "play", "row": 2, "col": 2}],
        "level": "az-iter001",
    },
)
assert resp.status_code == 200, resp.text
body = resp.json()
print("move:", body["move"])
print("win_rate:", round(body["win_rate"], 3), "| policy entries:", len(body["policy"]))
print("OK")
