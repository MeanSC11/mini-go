"""Go strength estimates: ELO <-> kyu/dan, win-rate -> rank, sims -> rank.

Two very different kinds of number live here, and we are careful to label which
is which when reporting:

* **ELO estimate** -- derived from internal self-evaluation between checkpoints.
  It is anchored arbitrarily (iteration 0 == ELO 0) so its *absolute* kyu/dan
  meaning rests on an assumption (see ``ANCHOR_KYU``). Good for tracking
  relative progress; only a rough guess of real-world rank.

* **Reference-calibrated** -- derived from games against an engine whose rank is
  known (e.g. GNU Go). This is the trustworthy number; it does not depend on the
  anchor assumption, only on the reference's assumed rank and the measured win
  rate.

Ranks are handled internally as a single continuous "rank number" g on the
go scale where ``... 2k=-1, 1k=0, 1d=1, 2d=2 ...`` (adjacent ranks = one stone).
"""

from __future__ import annotations

import math
from typing import Optional

# --- assumptions (documented; override via the CLI/config where it matters) ---

DEFAULT_ELO_PER_RANK = 100.0  # internal ELO per one-stone rank step (rough)
ANCHOR_ELO = 0.0              # internal ELO of the iteration-0 random network
ANCHOR_KYU = 30              # ...assumed to play around 30 kyu (raw beginner)


def kyu_dan_to_g(kyu: Optional[int] = None, dan: Optional[int] = None) -> float:
    """Rank number g for a kyu or dan rank (1k=0, 1d=1, 2k=-1, ...)."""
    if dan is not None:
        return float(dan)
    assert kyu is not None
    return float(1 - kyu)


def g_to_label(g: float) -> str:
    """Format a rank number as an approximate human label."""
    if g >= 1.0:
        dan = int(round(g))
        dan = max(1, min(dan, 9))
        return f"~{dan} dan"
    kyu = int(round(1.0 - g))
    kyu = max(1, min(kyu, 30))
    return f"~{kyu} kyu"


def elo_to_g(
    elo: float,
    anchor_elo: float = ANCHOR_ELO,
    anchor_kyu: int = ANCHOR_KYU,
    elo_per_rank: float = DEFAULT_ELO_PER_RANK,
) -> float:
    """Map an internal ELO to a rank number under the anchor assumption."""
    anchor_g = kyu_dan_to_g(kyu=anchor_kyu)
    return anchor_g + (elo - anchor_elo) / elo_per_rank


def elo_to_label(elo: float, **kwargs) -> str:
    """Approximate kyu/dan label for an internal ELO (anchor-dependent)."""
    return g_to_label(elo_to_g(elo, **kwargs))


def winrate_to_elo_diff(win_rate: float) -> float:
    """ELO difference implied by a win rate (inverse of the logistic curve).

    Positive means stronger than the opponent. Clamped away from 0/1 so a
    sweep/whitewash yields a finite (if large) estimate.
    """
    w = min(max(win_rate, 1e-3), 1.0 - 1e-3)
    return -400.0 * math.log10(1.0 / w - 1.0)


def reference_calibrated_g(
    reference_g: float,
    win_rate: float,
    elo_per_rank: float = DEFAULT_ELO_PER_RANK,
) -> float:
    """Rank number implied by ``win_rate`` against a reference of known rank."""
    return reference_g + winrate_to_elo_diff(win_rate) / elo_per_rank


# --- simulations -> rank, used by the resource assessment --------------------

# Heuristic: strength grows ~one stone per doubling of simulations/move beyond a
# small base. This is a DEFAULT curve; a real calibration table (produced by the
# benchmark against a reference engine) should override it when available.
_BASE_SIMS = 32.0
_STONES_PER_DOUBLING = 1.0


def sims_to_g(
    sims_per_move: float,
    base_g: float,
    base_sims: float = _BASE_SIMS,
    stones_per_doubling: float = _STONES_PER_DOUBLING,
) -> float:
    """Estimate rank number at ``sims_per_move`` given a base anchor.

    ``base_g`` is the rank at ``base_sims`` (e.g. from ELO estimate or a
    reference calibration). More simulations -> deeper search -> stronger play,
    *with the same network* (no quality change, just more thinking).
    """
    if sims_per_move <= 0:
        return base_g
    return base_g + stones_per_doubling * math.log2(max(sims_per_move, 1.0) / base_sims)


def parse_rank(text: str) -> float:
    """Parse '6k', '6 kyu', '2d', '2 dan' (case-insensitive) into a rank number."""
    s = text.strip().lower().replace(" ", "")
    for suffix, is_dan in (("kyu", False), ("k", False), ("dan", True), ("d", True)):
        if s.endswith(suffix):
            value = int(s[: -len(suffix)])
            return kyu_dan_to_g(dan=value) if is_dan else kyu_dan_to_g(kyu=value)
    raise ValueError(f"unrecognized rank: {text!r} (use e.g. '6k' or '2d')")
