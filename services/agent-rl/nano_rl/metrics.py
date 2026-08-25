"""evaluation metrics.

every definition here is stated explicitly, because most of them have a
flattering variant and a defensible one, and the difference is rarely visible
in a results table.

  sharpe        computed per EPISODE and annualised by the actual number of
                episodes per year, not per trade and not with an assumed 252.
                episodes are 15 minutes long, so there are 35,040 per year.
                annualising a per-trade sharpe by 252 is the standard way to
                report a number four times larger than the truth.

  max drawdown  on the cumulative pnl curve in dollars, not on returns, since
                the strategy has no meaningful capital base to divide by. a
                binary contract is fully collateralised at entry.

  hit rate      fraction of episodes with strictly positive pnl. episodes with
                exactly zero (the always-flat baseline) count as neither wins
                nor losses, and are reported separately, because counting them
                as wins would give a do-nothing policy a 100% hit rate.

  turnover      contracts traded per episode, normalised by max position. this
                is what the fee schedule actually bills.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# 15-minute episodes, running continuously
EPISODES_PER_YEAR = 365 * 24 * 4


@dataclass
class Metrics:
    """the full result row for one policy on one split."""

    n_episodes: int
    total_pnl: float
    mean_pnl: float
    std_pnl: float
    sharpe: float
    sharpe_annualised: float
    max_drawdown: float
    hit_rate: float
    zero_rate: float
    mean_trades: float
    turnover: float
    mean_fees: float

    def as_dict(self) -> dict:
        return asdict(self)


def sharpe_ratio(pnl: np.ndarray, annualise: bool = False) -> float:
    """mean over standard deviation of per-episode pnl.

    returns 0.0 rather than inf when the strategy never varies, which is the
    always-flat case. an infinite sharpe for a policy that earns nothing is
    technically defensible and practically absurd.
    """
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 2:
        return 0.0
    sd = pnl.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    s = float(pnl.mean() / sd)
    return s * np.sqrt(EPISODES_PER_YEAR) if annualise else s


def max_drawdown(pnl: np.ndarray) -> float:
    """largest peak-to-trough decline of cumulative pnl, in dollars.

    returned as a positive number. zero means the equity curve never declined.
    """
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) == 0:
        return 0.0
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(equity)
    return float(np.max(running_peak - equity))


def hit_rate(pnl: np.ndarray) -> tuple[float, float]:
    """(fraction strictly positive, fraction exactly zero).

    separating the two matters: a do-nothing policy produces all zeros, and
    folding those into either bucket misrepresents it.
    """
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) == 0:
        return 0.0, 0.0
    nonzero = ~np.isclose(pnl, 0.0)
    if nonzero.sum() == 0:
        return 0.0, 1.0
    return float((pnl > 0).sum() / len(pnl)), float((~nonzero).sum() / len(pnl))


def compute_metrics(
    pnl: np.ndarray,
    trades: np.ndarray,
    fees: np.ndarray,
    max_position: float = 100.0,
) -> Metrics:
    """assemble the full metric row for one policy."""
    pnl = np.asarray(pnl, dtype=float)
    trades = np.asarray(trades, dtype=float)
    fees = np.asarray(fees, dtype=float)

    hits, zeros = hit_rate(pnl)

    return Metrics(
        n_episodes=len(pnl),
        total_pnl=float(pnl.sum()),
        mean_pnl=float(pnl.mean()) if len(pnl) else 0.0,
        std_pnl=float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        sharpe=sharpe_ratio(pnl),
        sharpe_annualised=sharpe_ratio(pnl, annualise=True),
        max_drawdown=max_drawdown(pnl),
        hit_rate=hits,
        zero_rate=zeros,
        mean_trades=float(trades.mean()) if len(trades) else 0.0,
        turnover=float(trades.mean() * max_position) if len(trades) else 0.0,
        mean_fees=float(fees.mean()) if len(fees) else 0.0,
    )


def aggregate_across_seeds(rows: list[Metrics]) -> dict[str, tuple[float, float]]:
    """mean and std of each metric across seeds.

    the spec requires mean +/- std rather than a single run, and phase 3
    established why that is not a formality here: on signal-free data ppo's
    outcome is bimodal across seeds, so any single run misrepresents it.
    """
    if not rows:
        return {}
    keys = [k for k, v in rows[0].as_dict().items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.array([getattr(r, k) for r in rows], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def paired_bootstrap_p_value(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> float:
    """two-sided p-value for mean(a) - mean(b), by paired bootstrap.

    paired because both policies see the same episodes, so the episode-level
    settlement noise (which dominates everything here at a per-episode standard
    deviation near 50) is common to both and cancels.

    included because with a per-episode std of ~50 and differences of ~1, an
    unpaired eyeball comparison of means is worthless.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")

    diff = a - b
    observed = diff.mean()
    if np.allclose(diff, 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    centred = diff - observed
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = centred[idx].mean(axis=1)
    return float((np.abs(boot_means) >= abs(observed)).mean())
