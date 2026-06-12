"use client";

/**
 * SVG Go board. Pure presentation: renders the board strings provided by the
 * server and reports clicked intersections. No game rules live here.
 */

interface BoardProps {
  board: string[]; // rows of '.', 'B', 'W'
  size: number;
  lastMove?: { row: number; col: number } | null;
  heatmap?: Record<string, number> | null; // "row,col" -> probability
  disabled?: boolean;
  onPlay?: (row: number, col: number) => void;
}

const CELL = 40;
const MARGIN = 30;

function starPoints(size: number): [number, number][] {
  if (size === 9)
    return [
      [2, 2],
      [2, 6],
      [6, 2],
      [6, 6],
      [4, 4],
    ];
  if (size === 13)
    return [
      [3, 3],
      [3, 9],
      [9, 3],
      [9, 9],
      [6, 6],
    ];
  if (size === 19)
    return [3, 9, 15].flatMap((r) => [3, 9, 15].map((c) => [r, c] as [number, number]));
  return [];
}

export default function Board({
  board,
  size,
  lastMove,
  heatmap,
  disabled,
  onPlay,
}: BoardProps) {
  const extent = (size - 1) * CELL + 2 * MARGIN;
  const pos = (i: number) => MARGIN + i * CELL;
  const maxPolicy = heatmap
    ? Math.max(0.0001, ...Object.values(heatmap))
    : 1;

  return (
    <svg
      viewBox={`0 0 ${extent} ${extent}`}
      className="board"
      role="img"
      aria-label="Go board"
    >
      <rect width={extent} height={extent} fill="#deb068" rx={8} />
      {Array.from({ length: size }, (_, i) => (
        <g key={i} stroke="#5b4423" strokeWidth={1}>
          <line x1={pos(0)} y1={pos(i)} x2={pos(size - 1)} y2={pos(i)} />
          <line x1={pos(i)} y1={pos(0)} x2={pos(i)} y2={pos(size - 1)} />
        </g>
      ))}
      {starPoints(size).map(([r, c]) => (
        <circle key={`star-${r}-${c}`} cx={pos(c)} cy={pos(r)} r={4} fill="#5b4423" />
      ))}
      {heatmap &&
        Object.entries(heatmap).map(([key, prob]) => {
          if (key === "pass") return null;
          const [r, c] = key.split(",").map(Number);
          if (board[r]?.[c] !== ".") return null;
          return (
            <circle
              key={`heat-${key}`}
              cx={pos(c)}
              cy={pos(r)}
              r={CELL * 0.42}
              fill="#1d8a4b"
              opacity={0.15 + 0.65 * (prob / maxPolicy)}
            />
          );
        })}
      {board.map((row, r) =>
        row.split("").map((cell, c) => {
          if (cell === ".") return null;
          const isLast = lastMove && lastMove.row === r && lastMove.col === c;
          return (
            <g key={`stone-${r}-${c}`}>
              <circle
                cx={pos(c)}
                cy={pos(r)}
                r={CELL * 0.46}
                fill={cell === "B" ? "#1b1b1b" : "#f4f4f0"}
                stroke="#33301f"
                strokeWidth={0.8}
              />
              {isLast && (
                <circle
                  cx={pos(c)}
                  cy={pos(r)}
                  r={CELL * 0.2}
                  fill="none"
                  stroke={cell === "B" ? "#f4f4f0" : "#1b1b1b"}
                  strokeWidth={2}
                />
              )}
            </g>
          );
        })
      )}
      {!disabled &&
        onPlay &&
        board.map((row, r) =>
          row.split("").map((cell, c) =>
            cell === "." ? (
              <circle
                key={`hit-${r}-${c}`}
                cx={pos(c)}
                cy={pos(r)}
                r={CELL * 0.46}
                fill="transparent"
                style={{ cursor: "pointer" }}
                onClick={() => onPlay(r, c)}
              />
            ) : null
          )
        )}
    </svg>
  );
}
