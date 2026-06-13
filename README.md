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

The loop is the classic AlphaZero cycle: the **best** model plays games against
itself, those games train a **candidate**, and the candidate only replaces best
if it actually wins their match. Repeat — each pass the model gets a little
stronger and feeds itself harder games.

```
                         ┌──────────────────────────────────────────┐
                         │          ONE ITERATION (repeat ×N)         │
                         └──────────────────────────────────────────┘

   best.pt ──────┐
   (current      │   ┌─────────────────────┐   (state, policy π, outcome z) samples
    champion)    └──▶│  1. SELF-PLAY        │──────────────┐
                     │  games_per_iteration │              │
                     │  games, `workers`    │              ▼
                     │  parallel procs.     │     ┌────────────────────┐
                     │  PUCT MCTS +         │     │  2. REPLAY BUFFER   │
                     │  Dirichlet noise     │     │  buffer_size ring   │
                     └─────────────────────┘     │  persisted to disk  │
                                                 │  (runs/replay.npz)  │
                                                 └─────────┬──────────┘
                                                           │ sample
                          buffer < min_buffer_size?        │ batches
                          → skip training, play more       ▼
                                                 ┌────────────────────┐
                                                 │  3. TRAIN CANDIDATE │
                                                 │  train_steps SGD    │
                                                 │  loss = policy(CE)  │
                                                 │       + value(MSE)  │
                                                 └─────────┬──────────┘
                                                           │ candidate.pt
                                                           ▼
                                                 ┌────────────────────┐
                                  ┌──────────────│  4. EVALUATE        │
                                  │              │  candidate vs best  │
                                  │              │  eval_games match   │
                                  │              └─────────┬──────────┘
                       win_rate <  │                        │ win_rate ≥
                       promotion   │                        │ promotion_win_rate
                       _win_rate   │                        ▼
                                  │              ┌────────────────────┐
                          keep    │              │  5. PROMOTE         │
                          best,   │              │  candidate → best   │
                          candidate│             └─────────┬──────────┘
                          trains   │                        │
                          on       ▼                        ▼
                  ┌─────────────────────────────────────────────────┐
                  │  Always write checkpoints/iterNNN.pt             │
                  │  → appears as bot level `az-iterNNN` in the      │
                  │    inference service / lobby                     │
                  └─────────────────────────────────────────────────┘
                                  │
                                  └────────▶ next iteration uses the new best.pt
```

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

### Plateau-aware training (train for max strength)

The loop tracks an internal ELO (candidate vs best each iteration), logs it with
a rough rank estimate, and always keeps `best.pt` pointing at the strongest
network so far — you can pull `best.pt` and play/benchmark it mid-run without
stopping. Stopping is a three-layer policy that tries to push further before
giving up:

1. **detect** — no >`plateau_elo_threshold` ELO gain for `plateau_patience`
   iterations ⇒ suspect a plateau;
2. **shake** — lower the LR, raise self-play exploration (Dirichlet noise +
   opening temperature) and simulations for `shake_iterations`; if ELO climbs
   again, resume normal training;
3. **accept** — if the shake doesn't help, report the ceiling and stop.

A `max_hours` circuit breaker stops the run regardless (set it below your GPU
budget). State is checkpointed every iteration to `runs/<name>/training_state.json`,
so `azero-train --config ... --resume` continues a dropped run with ELO/plateau
state intact. `config.19x19.yaml` is set up for a max-strength run (effectively
uncapped iterations; `max_hours`/plateau decide when to stop).

## Serving & measuring strength

Serving is decoupled from training: play strength is governed by how many MCTS
**simulations** (or how much **time/move**) you give the bot, using the same
network. More thinking = stronger play, with no change to the model. At serve
time evaluations go through a transposition/eval cache and optional BF16/FP16
(`precision: auto`), and the search is batched/leaf-parallel like self-play.

The inference service reads `GOBOT_INF_AZ_SIMULATIONS` (sims/move) or
`GOBOT_INF_AZ_TIME_PER_MOVE` (seconds/move; overrides sims when set).

### What rank is this bot? (`azero-benchmark`)

```bash
# Accurate: calibrate against a reference engine of known rank (GTP).
azero-benchmark checkpoints/best.pt --sims 400 --games 30 \
    --reference gnugo --reference-rank 6k

# No reference engine installed -> rough ELO estimate (clearly labelled).
azero-benchmark checkpoints/best.pt --elo 1800
```

Two methods, with different trust levels:

- **Reference-calibrated** (accurate): the bot plays N games against a known
  engine (GNU Go ≈ 5–10 kyu, or Pachi) over GTP; the win rate is converted into
  a rank *relative to that reference*. GNU Go/Pachi are **optional** — install
  them only if you want this. Works on whatever board sizes the reference
  supports (GNU Go: 9/13/19).
- **ELO estimate** (rough fallback): when no reference engine is found, the
  internal self-evaluation ELO is mapped to kyu/dan under a stated **assumption**
  — iteration-0 (random net) ≈ 30 kyu, ~100 ELO per stone. Good for tracking
  *relative* progress between checkpoints; only a ballpark for real-world rank.

Output always states which method produced the number.

### What can *this machine* run? (`azero-assess`)

Run at serve time to size the speed/strength trade-off to your hardware. It
detects the GPU/CPU, micro-benchmarks MCTS throughput (~10–20 s), and estimates
the rank reachable at a given time/move (faster hardware → more sims → stronger):

```bash
azero-assess checkpoints/best.pt --time 5      # 5 s/move
```
```
Detected: GPU NVIDIA H100 80GB HBM3 (84.9 GB), 64 CPU cores
This machine runs ~38000 MCTS sims/second
At 5.0s/move -> ~190000 sims/move
Estimated playable strength: ~3 dan  (reference-calibrated)
```

Everything auto-detects hardware and falls back cleanly: GPU→CPU, BF16→FP16→FP32,
reference-calibrated→ELO estimate. None of this touches the training loop.

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
| inference | `GOBOT_INF_AZ_TIME_PER_MOVE` | _(unset; overrides sims when set)_ |
| inference | `GOBOT_INF_DEVICE` | `cpu` |
| web | `NEXT_PUBLIC_API_URL` (build-time) | `http://localhost:8000` |
