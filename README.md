# go-bot

A Go (Baduk) game with an AI bot trained via AlphaZero-style self-play, plus a Web UI for real play.

## Architecture

```
go-bot/
├── engine/        # Go rules (pure Python) — single source of truth for the rules
├── training/      # AlphaZero: ResNet + PUCT MCTS + self-play loop (PyTorch)
├── server/        # FastAPI game server: REST + WebSocket + PostgreSQL
├── inference/     # bot inference service (random / pure-MCTS / AlphaZero checkpoints)
└── web/           # Next.js frontend (SVG board)
```

- **engine** (`goengine`): 9x9/13x13/19x19 boards, captures, suicide forbidden,
  positional superko, Chinese scoring (komi 7.5), two consecutive passes end the
  game, SGF export/import — shared by every service; the frontend only renders
  and sends intents. Every move is validated on the server using the engine.
- **server**: create/join games via REST, play moves over WebSocket in real
  time, persist every game (with SGF) to PostgreSQL, ELO rating, request
  analysis (win rate + policy heatmap) from the inference service
- **inference**: separate HTTP service process holding the models — strength
  levels: `random` → `mcts-100/300/800` (pure UCT) → `az-iterNNN` (training
  checkpoints; higher iterations are stronger)
- **training**: self-play → replay buffer → train → evaluate against the
  previous model → promote on ≥ 55% win rate — TensorBoard logging, checkpoint
  every iteration

## Quick start (Docker)

```bash
cd go-bot
docker compose up --build
```

- Web UI: http://localhost:3000
- Game server API: http://localhost:8000 (OpenAPI docs at `/docs`)
- Inference service: http://localhost:8001
- PostgreSQL: localhost:5432 (`gobot`/`gobot`)

Trained checkpoints in `training/checkpoints/` are mounted into the inference
service automatically and show up as bot levels in the lobby.

## Dev environment (without Docker)

Requires Python ≥ 3.10, Node ≥ 20, and PostgreSQL (or let the server use a
temporary SQLite database via `GOBOT_DATABASE_URL=sqlite+aiosqlite:///./dev.db`)

```bash
cd go-bot
python -m venv .venv
.venv/Scripts/activate           # Windows  (Linux/macOS: source .venv/bin/activate)

pip install -e engine[dev]
pip install -e server[dev]
pip install -e inference[dev]
pip install -e training[dev] --extra-index-url https://download.pytorch.org/whl/cpu
```

Run each service (separate terminals):

```bash
# 1) database (if Postgres is not installed locally)
docker compose up db

# 2) game server (port 8000)
cd server && uvicorn app.main:app --reload --port 8000

# 3) inference service (port 8001)
cd inference && uvicorn app.main:app --reload --port 8001

# 4) web (port 3000)
cd web && npm install && npm run dev
```

### Run tests

```bash
cd engine    && python -m pytest      # rules: capture, ko, snapback, scoring, SGF
cd server    && python -m pytest      # REST + WebSocket + persistence
cd inference && python -m pytest      # UCT MCTS + HTTP API
cd training  && python -m pytest      # features, network, PUCT, self-play, checkpoints
```

## Training

All hyperparameters live in [training/config.yaml](training/config.yaml)
(board size, blocks/filters, simulations, buffer, learning rate, promotion
threshold, etc.)

```bash
cd training
azero-train --config config.yaml            # start training (auto-falls back to CPU without CUDA)
azero-train --config config.yaml --resume   # resume from best.pt + existing replay buffer

tensorboard --logdir runs                   # view loss / win rate / promotions
```

Each iteration:
1. self-plays `games_per_iteration` games with the current best model
   (in parallel across `workers` processes)
2. stores `(state, policy, outcome)` in the replay buffer (persisted to disk)
3. trains a candidate for `train_steps_per_iteration` steps
4. pits the candidate against best for `eval_games` games — it must win
   ≥ `promotion_win_rate` to be promoted as the new best
5. writes checkpoint `checkpoints/iterNNN.pt` (becomes bot level `az-iterNNN`
   in the inference service as soon as it is mounted)

Generate self-play data separately, or pit two checkpoints against each other:

```bash
azero-selfplay --config config.yaml --checkpoint checkpoints/best.pt --games 100 --out data.npz
azero-evaluate checkpoints/iter010.pt checkpoints/iter002.pt --games 20
```

GPU: set `device: cuda` in the config (already the default — falls back to CPU
automatically). For Docker, switch the base image in `training/Dockerfile` to a
CUDA build as noted in the file's comments.

## Gameplay features

- Play human vs human (create a room and have someone join) or human vs bot
- Choose bot level: random / pure MCTS (100/300/800 sims) / AlphaZero checkpoints
- Click-to-place SVG board, captured-stone counts, pass/resign buttons, final score at game end
- History slider to step through every move (server replays via the engine — no rule logic in the frontend)
- Analyze button + heatmap toggle: shows the bot's win rate and policy on the board
- SGF download for every game
- ELO rating for players who register a name (leaderboard at `GET /api/users`)

## Environment variables

| Service | Variable | Default |
|---|---|---|
| server | `GOBOT_DATABASE_URL` | `postgresql+asyncpg://gobot:gobot@localhost:5432/gobot` |
| server | `GOBOT_INFERENCE_URL` | `http://localhost:8001` |
| server | `GOBOT_CORS_ORIGINS` | `["http://localhost:3000"]` |
| inference | `GOBOT_INF_CHECKPOINT_DIR` | `checkpoints` |
| inference | `GOBOT_INF_AZ_SIMULATIONS` | `160` |
| inference | `GOBOT_INF_DEVICE` | `cpu` |
| web | `NEXT_PUBLIC_API_URL` (build-time) | `http://localhost:8000` |
