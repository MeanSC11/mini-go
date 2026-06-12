"""ELO rating updates."""

from __future__ import annotations

from typing import Tuple

K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score of player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update(
    rating_a: float, rating_b: float, score_a: float, k: float = K_FACTOR
) -> Tuple[float, float]:
    """Return updated ``(rating_a, rating_b)``.

    ``score_a`` is 1.0 for an A win, 0.0 for a loss, 0.5 for a draw.
    """
    ea = expected_score(rating_a, rating_b)
    delta = k * (score_a - ea)
    return rating_a + delta, rating_b - delta
