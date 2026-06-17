"""Pure Monte-Carlo Tree Search with UCT (no neural network).

Phase-3 bot: uniform-random playouts scored with Chinese rules. Strength is
controlled by the simulation budget per move.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from goengine import Color, Game, Move, Point, candidate_moves, is_eye_fill
from goengine.scoring import area_score

UCT_C = 1.4


def is_simple_eye(game: Game, point: Point, color: Color) -> bool:
    """Heuristic single-point eye check (shared engine helper).

    A point is treated as an eye if every neighbor is ``color`` and the
    diagonals are sufficiently controlled (all in the corner/edge case, at
    most one enemy diagonal in the interior).
    """
    return is_eye_fill(game.board, point, color)


def _random_playout_move(game: Game, rng: random.Random) -> Move:
    """Pick a random legal move that is not an own-eye fill; else pass."""
    empties = list(game.board.empty_points())
    rng.shuffle(empties)
    color = game.current_player
    for point in empties:
        if is_simple_eye(game, point, color):
            continue
        move = Move.play(*point)
        if game.is_legal(move):
            return move
    return Move.pass_turn()


def rollout_winner(game: Game, rng: random.Random) -> Color:
    """Play uniformly random moves to the end and return the area winner."""
    playout = game.copy()
    max_moves = 2 * playout.board_size * playout.board_size
    for _ in range(max_moves):
        if playout.is_over:
            break
        playout.play(_random_playout_move(playout, rng))
    if playout.result is not None and playout.result.winner is not Color.EMPTY:
        return playout.result.winner
    black, white = area_score(playout.board)
    return Color.BLACK if black > white + playout.komi else Color.WHITE


class _Node:
    """One MCTS tree node. ``player`` is the side that made ``move``."""

    __slots__ = ("parent", "move", "player", "children", "visits", "wins", "untried")

    def __init__(
        self, parent: Optional["_Node"], move: Optional[Move], player: Color, game: Game
    ) -> None:
        self.parent = parent
        self.move = move
        self.player = player
        self.children: List[_Node] = []
        self.visits = 0
        self.wins = 0.0
        # Search over real moves only — exclude own-eye fills so the bot passes
        # once the game is decided instead of filling its own territory.
        self.untried: List[Move] = candidate_moves(game) if not game.is_over else []

    def ucb_child(self) -> "_Node":
        log_n = math.log(self.visits)
        return max(
            self.children,
            key=lambda c: c.wins / c.visits + UCT_C * math.sqrt(log_n / c.visits),
        )


class UctBot:
    """UCT MCTS bot with a fixed simulation budget per move."""

    def __init__(self, simulations: int = 300, seed: Optional[int] = None) -> None:
        self.simulations = simulations
        self.rng = random.Random(seed)

    def search(self, game: Game) -> Tuple[Move, float, Dict[str, float]]:
        """Run MCTS from ``game`` and return ``(move, win_rate, policy)``.

        ``win_rate`` is the estimated win probability for the side to move;
        ``policy`` maps ``"row,col"`` (or ``"pass"``) to visit fractions.
        """
        root_player = game.current_player
        root = _Node(None, None, root_player.opponent, game)
        if not root.untried:
            return Move.pass_turn(), 0.5, {}
        for _ in range(self.simulations):
            node = root
            sim = game.copy()
            # Selection
            while not node.untried and node.children:
                node = node.ucb_child()
                sim.play(node.move)  # type: ignore[arg-type]
            # Expansion
            if node.untried and not sim.is_over:
                move = node.untried.pop(self.rng.randrange(len(node.untried)))
                mover = sim.current_player
                sim.play(move)
                child = _Node(node, move, mover, sim)
                node.children.append(child)
                node = child
            # Simulation
            winner = rollout_winner(sim, self.rng)
            # Backpropagation
            while node is not None:
                node.visits += 1
                if winner is node.player:
                    node.wins += 1.0
                node = node.parent  # type: ignore[assignment]

        best = max(root.children, key=lambda c: c.visits)
        # Prefer passing when playing on does not improve the position: in a
        # decided game every move scores the same, so the bot should stop
        # filling instead of playing pointless stones.
        pass_child = next((c for c in root.children if c.move and c.move.is_pass), None)
        if pass_child is not None and pass_child.visits > 0 and best.visits > 0:
            if pass_child.wins / pass_child.visits >= best.wins / best.visits:
                best = pass_child
        win_rate = best.wins / best.visits if best.visits else 0.5
        total_visits = sum(c.visits for c in root.children)
        policy: Dict[str, float] = {}
        for child in root.children:
            move = child.move
            assert move is not None
            key = "pass" if move.is_pass else f"{move.point[0]},{move.point[1]}"  # type: ignore[index]
            policy[key] = child.visits / total_visits
        assert best.move is not None
        return best.move, win_rate, policy
