"use client";

/** Game page: live board, controls, history scrubbing, optional bot heatmap. */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import Board from "@/components/Board";
import {
  HistoryResponse,
  getAnalysis,
  getHistory,
  getSgf,
} from "@/lib/api";
import { useGameSocket } from "@/lib/useGameSocket";

interface Credentials {
  color: "black" | "white";
  token: string;
}

export default function GamePage() {
  const params = useParams<{ id: string }>();
  const gameId = params.id;

  const [creds, setCreds] = useState<Credentials | null>(null);
  const [credsLoaded, setCredsLoaded] = useState(false);
  useEffect(() => {
    const raw = sessionStorage.getItem(`game:${gameId}`);
    if (raw) setCreds(JSON.parse(raw) as Credentials);
    setCredsLoaded(true);
  }, [gameId]);

  const { state, analysis, error, connected, sendMove, setAnalysis } =
    useGameSocket(gameId, credsLoaded ? creds?.token ?? null : null);

  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [viewIndex, setViewIndex] = useState<number | null>(null); // null = live
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [analysisBusy, setAnalysisBusy] = useState(false);

  const moveCount = state?.moves.length ?? 0;

  // Refresh history whenever the move list grows (cheap on small boards).
  useEffect(() => {
    if (!state) return;
    getHistory(gameId).then(setHistory).catch(() => undefined);
  }, [gameId, moveCount, state?.status]);  // eslint-disable-line react-hooks/exhaustive-deps

  // New moves invalidate any displayed analysis.
  useEffect(() => {
    if (!state?.mode || state.mode !== "hvb") setAnalysis(null);
  }, [moveCount]); // eslint-disable-line react-hooks/exhaustive-deps

  const isLive = viewIndex === null;
  const shownBoard = useMemo(() => {
    if (!state) return null;
    if (isLive || !history) return state.board;
    return history.boards[viewIndex!] ?? state.board;
  }, [state, history, viewIndex, isLive]);

  const lastMove = useMemo(() => {
    if (!state) return null;
    const idx = isLive ? state.moves.length - 1 : viewIndex! - 1;
    const move = state.moves[idx];
    if (move && move.type === "play" && move.row != null && move.col != null) {
      return { row: move.row, col: move.col };
    }
    return null;
  }, [state, viewIndex, isLive]);

  const myTurn =
    state?.status === "playing" &&
    creds != null &&
    state.current_player === creds.color;

  const handlePlay = useCallback(
    (row: number, col: number) => {
      if (isLive && myTurn) sendMove({ type: "play", row, col });
    },
    [isLive, myTurn, sendMove]
  );

  const handleAnalyze = async () => {
    setAnalysisBusy(true);
    try {
      setAnalysis(await getAnalysis(gameId));
      setShowHeatmap(true);
    } catch {
      // inference service not running; ignore
    } finally {
      setAnalysisBusy(false);
    }
  };

  const downloadSgf = async () => {
    const { sgf } = await getSgf(gameId);
    const blob = new Blob([sgf], { type: "application/x-go-sgf" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${gameId}.sgf`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (!state || !shownBoard) {
    return <p className="muted">Loading game…</p>;
  }

  const winRatePercent =
    analysis?.win_rate != null
      ? Math.round(analysis.win_rate * 1000) / 10
      : null;

  return (
    <div className="game-layout">
      <div>
        {state.result && (
          <div className="result-banner">
            Result: {state.result}
            {state.score_black != null && state.score_white != null && (
              <span className="muted">
                {" "}
                (B {state.score_black} : W {state.score_white})
              </span>
            )}
          </div>
        )}
        <Board
          board={shownBoard}
          size={state.board_size}
          lastMove={lastMove}
          heatmap={showHeatmap && isLive ? analysis?.policy ?? null : null}
          disabled={!isLive || !myTurn}
          onPlay={handlePlay}
        />
        {history && history.boards.length > 1 && (
          <div className="card" style={{ marginTop: 12 }}>
            <div className="panel-row">
              <span className="muted">
                Move {isLive ? history.boards.length - 1 : viewIndex} /{" "}
                {history.boards.length - 1}
              </span>
              {!isLive && (
                <button className="secondary" onClick={() => setViewIndex(null)}>
                  Back to live
                </button>
              )}
            </div>
            <input
              type="range"
              className="history-slider"
              min={0}
              max={history.boards.length - 1}
              value={isLive ? history.boards.length - 1 : viewIndex!}
              onChange={(e) => {
                const value = Number(e.target.value);
                setViewIndex(
                  value === history.boards.length - 1 ? null : value
                );
              }}
            />
          </div>
        )}
      </div>

      <div className="card">
        <div className="panel-row">
          <span className={`badge ${state.status === "playing" ? "turn" : ""}`}>
            {state.status === "waiting"
              ? "Waiting for opponent"
              : state.status === "finished"
              ? "Finished"
              : `${state.current_player} to play`}
          </span>
          <span className="muted">{connected ? "● live" : "○ reconnecting"}</span>
        </div>
        {creds && (
          <p className="muted">
            You play <strong>{creds.color}</strong>
            {state.mode === "hvb" && ` vs bot (${state.bot_level})`}
          </p>
        )}
        {!creds && <p className="muted">Spectating</p>}
        <div className="panel-row">
          <span>⚫ captured: {state.captures_black}</span>
          <span>⚪ captured: {state.captures_white}</span>
        </div>
        <div className="panel-row">
          <span className="muted">
            {state.board_size}×{state.board_size} · komi {state.komi}
          </span>
        </div>

        {winRatePercent != null && (
          <div>
            <div className="panel-row">
              <span className="muted">Black win rate: {winRatePercent}%</span>
            </div>
            <div className="winrate-bar">
              <div style={{ width: `${winRatePercent}%` }} />
            </div>
          </div>
        )}

        <div className="panel-row" style={{ marginTop: 16, gap: 8 }}>
          <button
            className="secondary"
            disabled={!myTurn || !isLive}
            onClick={() => sendMove({ type: "pass" })}
          >
            Pass
          </button>
          <button
            className="danger"
            disabled={!creds || state.status !== "playing"}
            onClick={() => sendMove({ type: "resign" })}
          >
            Resign
          </button>
        </div>
        {state.mode === "hvb" && state.status === "playing" && (
          <div className="panel-row" style={{ gap: 8 }}>
            <button
              className="secondary"
              onClick={handleAnalyze}
              disabled={analysisBusy}
            >
              {analysisBusy ? "Analyzing…" : "Analyze position"}
            </button>
            <label className="muted">
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />{" "}
              heatmap
            </label>
          </div>
        )}
        <div className="panel-row">
          <button className="secondary" onClick={downloadSgf}>
            Download SGF
          </button>
        </div>
        <p className="error">{error}</p>
      </div>
    </div>
  );
}
