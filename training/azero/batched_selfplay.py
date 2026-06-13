"""In-process self-play: drive the engine with a local network as evaluator.

Used when ``workers == 1`` (a single GPU box, no worker processes). The heavy
lifting lives in :mod:`azero.selfplay_engine`; here ``evaluate`` is just the
local network's batched forward pass.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from azero.config import Config
from azero.network import PolicyValueNet
from azero.replay import Sample
from azero.selfplay_engine import SelfPlayStats, run_self_play


def generate_games_batched(
    net: PolicyValueNet,
    config: Config,
    total_games: int,
    autocast_dtype: Any = None,
    stats: Optional[SelfPlayStats] = None,
) -> List[Sample]:
    """Generate ``total_games`` games in-process using ``net`` for evaluation."""

    def evaluate(planes: np.ndarray):
        return net.predict_many(planes, autocast_dtype)

    return run_self_play(
        evaluate, config, total_games, config.selfplay_concurrent_games, stats
    )
