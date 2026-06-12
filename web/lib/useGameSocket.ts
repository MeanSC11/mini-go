"use client";

/** WebSocket hook: subscribes to game state and exposes a move sender. */

import { useCallback, useEffect, useRef, useState } from "react";
import { Analysis, GameState, MovePayload, WS_URL } from "@/lib/api";

interface SocketMessage {
  type: "state" | "analysis" | "error" | "pong";
  state?: GameState;
  analysis?: Analysis;
  detail?: string;
}

export function useGameSocket(gameId: string, token: string | null) {
  const [state, setState] = useState<GameState | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const url = `${WS_URL}/ws/games/${gameId}${token ? `?token=${token}` : ""}`;
    let closed = false;
    let socket: WebSocket;

    const connect = () => {
      socket = new WebSocket(url);
      socketRef.current = socket;
      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        const message: SocketMessage = JSON.parse(event.data);
        if (message.type === "state" && message.state) {
          setState(message.state);
          setError(null);
        } else if (message.type === "analysis" && message.analysis) {
          setAnalysis(message.analysis);
        } else if (message.type === "error") {
          setError(message.detail ?? "unknown error");
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      socket.close();
    };
  }, [gameId, token]);

  const sendMove = useCallback((move: MovePayload) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "move", move }));
    }
  }, []);

  return { state, analysis, error, connected, sendMove, setAnalysis };
}
