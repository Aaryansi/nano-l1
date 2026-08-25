"""synthetic fixtures for env tests.

deliberately synthetic rather than sampled from the real corpus: a unit test
that depends on downloaded data cannot run on a clean checkout, and a test
whose expected values were read off real data tends to encode whatever bug
produced them. these fixtures have hand-computable properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.env.binary_market import EpisodeBatch


def make_batch(
    n_episodes: int = 4,
    n_steps: int = 10,
    spread: float = 0.02,
    mid_path: np.ndarray | None = None,
    settlement: np.ndarray | None = None,
    step_seconds: float = 60.0,
) -> EpisodeBatch:
    """build a small corpus with an exactly known mid-price path."""
    if mid_path is None:
        # flat at 0.50 so that any pnl must come from costs, not price moves
        mid_path = np.full((n_episodes, n_steps), 0.50, dtype=np.float32)
    mid = np.asarray(mid_path, dtype=np.float32)

    half = spread / 2.0
    bid = (mid - half).astype(np.float32)
    ask = (mid + half).astype(np.float32)

    if settlement is None:
        settlement = np.zeros(n_episodes, dtype=np.float32)

    t_sec = np.tile(
        np.arange(1, n_steps + 1, dtype=np.float32) * step_seconds, (n_episodes, 1)
    )

    return EpisodeBatch(
        bid=bid,
        ask=ask,
        last_price=mid.copy(),
        volume=np.full((n_episodes, n_steps), 1000.0, dtype=np.float32),
        staleness=np.zeros((n_episodes, n_steps), dtype=np.float32),
        flow_imbalance=np.zeros((n_episodes, n_steps), dtype=np.float32),
        t_sec=t_sec,
        settlement=np.asarray(settlement, dtype=np.float32),
        open_epoch=np.arange(n_episodes, dtype=np.float64) * 900.0,
    )


@pytest.fixture
def flat_batch() -> EpisodeBatch:
    """price pinned at 0.50 throughout, settles to 0."""
    return make_batch()


@pytest.fixture
def rising_batch() -> EpisodeBatch:
    """mid rises 0.40 -> 0.85 and settles yes; a profitable long exists."""
    n_ep, n_steps = 2, 10
    path = np.tile(np.linspace(0.40, 0.85, n_steps, dtype=np.float32), (n_ep, 1))
    return make_batch(
        n_episodes=n_ep,
        n_steps=n_steps,
        mid_path=path,
        settlement=np.ones(n_ep, dtype=np.float32),
    )
