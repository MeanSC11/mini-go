"use client";

/** Lobby: create a game (vs human or bot) or join a waiting game. */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  BotLevel,
  GameSummary,
  createGame,
  getBotLevels,
  joinGame,
  listGames,
} from "@/lib/api";

function storeCredentials(gameId: string, color: string, token: string): void {
  sessionStorage.setItem(`game:${gameId}`, JSON.stringify({ color, token }));
}

export default function Lobby() {
  const router = useRouter();
  const [boardSize, setBoardSize] = useState(9);
  const [mode, setMode] = useState<"hvh" | "hvb">("hvb");
  const [botLevel, setBotLevel] = useState("random");
  const [creatorColor, setCreatorColor] = useState<"black" | "white">("black");
  const [levels, setLevels] = useState<BotLevel[]>([]);
  const [waiting, setWaiting] = useState<GameSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getBotLevels()
      .then((r) => setLevels(r.levels))
      .catch(() => setLevels([{ name: "random", description: "Random moves" }]));
    const refresh = () =>
      listGames("waiting").then(setWaiting).catch(() => setWaiting([]));
    refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, []);

  const handleCreate = async () => {
    setBusy(true);
    setError(null);
    try {
      const creds = await createGame({
        board_size: boardSize,
        mode,
        bot_level: mode === "hvb" ? botLevel : undefined,
        creator_color: creatorColor,
      });
      storeCredentials(creds.game_id, creds.color, creds.token);
      router.push(`/game/${creds.game_id}`);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  const handleJoin = async (gameId: string) => {
    try {
      const creds = await joinGame(gameId);
      storeCredentials(creds.game_id, creds.color, creds.token);
      router.push(`/game/${gameId}`);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="lobby">
      <div className="card">
        <h2>New game</h2>
        <div className="field">
          <label>Board size</label>
          <select
            value={boardSize}
            onChange={(e) => setBoardSize(Number(e.target.value))}
          >
            <option value={9}>9 × 9</option>
            <option value={13}>13 × 13</option>
            <option value={19}>19 × 19</option>
          </select>
        </div>
        <div className="field">
          <label>Opponent</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "hvh" | "hvb")}
          >
            <option value="hvb">Bot</option>
            <option value="hvh">Human</option>
          </select>
        </div>
        {mode === "hvb" && (
          <div className="field">
            <label>Bot level</label>
            <select value={botLevel} onChange={(e) => setBotLevel(e.target.value)}>
              {levels.map((l) => (
                <option key={l.name} value={l.name}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
        )}
        <div className="field">
          <label>Your color</label>
          <select
            value={creatorColor}
            onChange={(e) => setCreatorColor(e.target.value as "black" | "white")}
          >
            <option value="black">Black (first)</option>
            <option value="white">White</option>
          </select>
        </div>
        <button onClick={handleCreate} disabled={busy}>
          {busy ? "Creating…" : "Start game"}
        </button>
        <p className="error">{error}</p>
      </div>

      <div className="card">
        <h2>Open games (waiting for opponent)</h2>
        {waiting.length === 0 && <p className="muted">No open games right now.</p>}
        <ul className="game-list">
          {waiting.map((g) => (
            <li key={g.game_id}>
              <span>
                {g.board_size}×{g.board_size} · {g.mode}
              </span>
              <button className="secondary" onClick={() => handleJoin(g.game_id)}>
                Join
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
