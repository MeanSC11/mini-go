/** REST client for the go-bot game server. */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_URL = API_URL.replace(/^http/, "ws");

export interface MovePayload {
  type: "play" | "pass" | "resign";
  row?: number | null;
  col?: number | null;
}

export interface GameState {
  game_id: string;
  board_size: number;
  komi: number;
  mode: string;
  bot_level: string | null;
  status: "waiting" | "playing" | "finished";
  board: string[];
  current_player: "black" | "white";
  captures_black: number;
  captures_white: number;
  moves: MovePayload[];
  result: string | null;
  score_black: number | null;
  score_white: number | null;
}

export interface PlayerCredentials {
  game_id: string;
  color: "black" | "white";
  token: string;
}

export interface GameSummary {
  game_id: string;
  board_size: number;
  mode: string;
  bot_level: string | null;
  status: string;
  result: string | null;
  created_at: string | null;
}

export interface HistoryResponse {
  boards: string[][];
  captures: [number, number][];
  moves: MovePayload[];
}

export interface Analysis {
  win_rate: number | null;
  policy: Record<string, number>;
}

export interface BotLevel {
  name: string;
  description: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status}: ${body}`);
  }
  return response.json() as Promise<T>;
}

export function createGame(options: {
  board_size: number;
  komi?: number;
  mode: "hvh" | "hvb";
  bot_level?: string;
  user_id?: string;
  creator_color?: "black" | "white";
}): Promise<PlayerCredentials> {
  return request("/api/games", {
    method: "POST",
    body: JSON.stringify(options),
  });
}

export function joinGame(
  gameId: string,
  userId?: string
): Promise<PlayerCredentials> {
  return request(`/api/games/${gameId}/join`, {
    method: "POST",
    body: JSON.stringify({ user_id: userId ?? null }),
  });
}

export function getGame(gameId: string): Promise<GameState> {
  return request(`/api/games/${gameId}`);
}

export function listGames(status?: string): Promise<GameSummary[]> {
  const suffix = status ? `?status=${status}` : "";
  return request(`/api/games${suffix}`);
}

export function getHistory(gameId: string): Promise<HistoryResponse> {
  return request(`/api/games/${gameId}/history`);
}

export function getSgf(gameId: string): Promise<{ sgf: string }> {
  return request(`/api/games/${gameId}/sgf`);
}

export function getAnalysis(gameId: string): Promise<Analysis> {
  return request(`/api/games/${gameId}/analysis`, { method: "POST" });
}

export function getBotLevels(): Promise<{ levels: BotLevel[] }> {
  return request("/api/bots/levels");
}

export function createUser(
  name: string
): Promise<{ id: string; name: string; elo: number; games_played: number }> {
  return request("/api/users", { method: "POST", body: JSON.stringify({ name }) });
}
