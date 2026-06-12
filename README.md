# go-bot

เกมหมากล้อม (Go/Baduk) พร้อม AI bot ที่เทรนด้วย self-play แบบ AlphaZero และ Web UI สำหรับเล่นจริง

## Architecture

```
go-bot/
├── engine/        # กติกาหมากล้อม (pure Python) — single source of truth ของกติกา
├── training/      # AlphaZero: ResNet + PUCT MCTS + self-play loop (PyTorch)
├── server/        # FastAPI game server: REST + WebSocket + PostgreSQL
├── inference/     # bot inference service (random / pure-MCTS / AlphaZero checkpoints)
└── web/           # Next.js frontend (SVG board)
```

- **engine** (`goengine`): กระดาน 9x9/13x13/19x19, จับกิน, ห้าม suicide,
  positional superko, นับคะแนนแบบ Chinese (komi 7.5), pass 2 ครั้งจบเกม,
  export/import SGF — ใช้ร่วมกันทุก service; frontend แค่ render และส่ง intent
  ทุกตาเดินถูก validate ที่ server ด้วย engine
- **server**: สร้าง/join เกมผ่าน REST, เดินหมากผ่าน WebSocket แบบ real-time,
  เก็บทุกเกม (พร้อม SGF) ลง PostgreSQL, ELO rating, ขอ analysis
  (win rate + policy heatmap) จาก inference service
- **inference**: HTTP service แยก process ถือ model — ระดับความเก่ง:
  `random` → `mcts-100/300/800` (pure UCT) → `az-iterNNN` (checkpoint จากการเทรน
  ยิ่ง iteration สูงยิ่งเก่ง)
- **training**: self-play → replay buffer → train → evaluate กับ model เก่า →
  promote เมื่อชนะ ≥ 55% — log ด้วย TensorBoard, checkpoint ทุก iteration

## Quick start (Docker)

```bash
cd go-bot
docker compose up --build
```

- Web UI: http://localhost:3000
- Game server API: http://localhost:8000 (OpenAPI docs ที่ `/docs`)
- Inference service: http://localhost:8001
- PostgreSQL: localhost:5432 (`gobot`/`gobot`)

Checkpoints ที่เทรนแล้วใน `training/checkpoints/` จะถูก mount เข้า inference
service อัตโนมัติ และโผล่เป็นระดับ bot ในหน้า lobby

## Dev environment (ไม่ใช้ Docker)

ต้องมี Python ≥ 3.10, Node ≥ 20 และ PostgreSQL (หรือปล่อยให้ server ใช้
SQLite ชั่วคราวผ่าน `GOBOT_DATABASE_URL=sqlite+aiosqlite:///./dev.db`)

```bash
cd go-bot
python -m venv .venv
.venv/Scripts/activate           # Windows  (Linux/macOS: source .venv/bin/activate)

pip install -e engine[dev]
pip install -e server[dev]
pip install -e inference[dev]
pip install -e training[dev] --extra-index-url https://download.pytorch.org/whl/cpu
```

รันแต่ละ service (คนละ terminal):

```bash
# 1) database (ถ้าไม่มี Postgres ติดตั้งไว้)
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
cd engine    && python -m pytest      # กติกา: capture, ko, snapback, scoring, SGF
cd server    && python -m pytest      # REST + WebSocket + persistence
cd inference && python -m pytest      # UCT MCTS + HTTP API
cd training  && python -m pytest      # features, network, PUCT, self-play, checkpoints
```

## Training

Hyperparameters ทั้งหมดอยู่ใน [training/config.yaml](training/config.yaml)
(board size, blocks/filters, simulations, buffer, learning rate, promotion
threshold ฯลฯ)

```bash
cd training
azero-train --config config.yaml            # เริ่มเทรน (auto-fallback เป็น CPU ถ้าไม่มี CUDA)
azero-train --config config.yaml --resume   # เทรนต่อจาก best.pt + replay buffer เดิม

tensorboard --logdir runs                   # ดู loss / win rate / promotions
```

แต่ละ iteration จะ:
1. self-play `games_per_iteration` เกมด้วย model best ปัจจุบัน
   (ขนานกัน `workers` process)
2. เก็บ `(state, policy, outcome)` ลง replay buffer (persist ลงดิสก์)
3. เทรน candidate `train_steps_per_iteration` steps
4. ประกบ candidate กับ best `eval_games` เกม — ชนะ ≥ `promotion_win_rate`
   ถึง promote เป็น best ใหม่
5. เขียน checkpoint `checkpoints/iterNNN.pt` (กลายเป็นระดับ bot `az-iterNNN`
   ใน inference service ทันทีที่ mount)

สร้าง self-play data แยก หรือประกบสอง checkpoint เอง:

```bash
azero-selfplay --config config.yaml --checkpoint checkpoints/best.pt --games 100 --out data.npz
azero-evaluate checkpoints/iter010.pt checkpoints/iter002.pt --games 20
```

GPU: ตั้ง `device: cuda` ใน config (default อยู่แล้ว — fallback เป็น CPU
อัตโนมัติ) สำหรับ Docker ให้เปลี่ยน base image ใน `training/Dockerfile`
เป็น CUDA build ตามคอมเมนต์ในไฟล์

## Gameplay features

- เล่น human vs human (สร้างห้องแล้วให้อีกคน join) หรือ human vs bot
- เลือกระดับ bot: random / pure MCTS (100/300/800 sims) / AlphaZero checkpoints
- กระดาน SVG คลิกวาง, แสดง captured stones, ปุ่ม pass/resign, ผลคะแนนตอนจบ
- History slider ย้อนดูทุกตาเดิน (server replay ผ่าน engine — ไม่มี rule logic ใน frontend)
- ปุ่ม Analyze + heatmap toggle: แสดง win rate และ policy ของ bot บนกระดาน
- ดาวน์โหลด SGF ได้ทุกเกม
- ELO rating สำหรับผู้เล่นที่ลงทะเบียนชื่อ (leaderboard ที่ `GET /api/users`)

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
