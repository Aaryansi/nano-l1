"""Settlement permutations and diagnostics for a changed decision problem.

Global permutation preserves feature arrays and the marginal settlement rate,
while changing their relationship to settlement and to rewards. Price remains
reward-relevant through acquisition cost even when it no longer predicts the
permuted settlement. This is not an efficient-market null or an intervention
that changes only information available to the agent.

Within-price-bucket permutation is an additional descriptive control. A valid
conditional randomization test would need justified exchangeability within its
conditioning structure; feature preservation alone does not supply that.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nano_rl.env.binary_market import EpisodeBatch


def permute_outcomes(batch: EpisodeBatch, seed: int = 0) -> EpisodeBatch:
    """shuffle settlements across episodes, leaving observations identical.

    the cached market-feature block is carried over deliberately rather than
    invalidated. it does not depend on settlement, so recomputing it would cost
    time and produce the same array; passing it through also makes the
    invariant explicit, and tests assert the observations really are unchanged.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(batch))
    return replace(
        batch,
        settlement=np.asarray(batch.settlement)[order].copy(),
        _market_features=batch.market_features(),
    )


def outcome_rate(batch: EpisodeBatch) -> float:
    """fraction of episodes settling to 1, which permutation must preserve."""
    return float(np.mean(np.asarray(batch.settlement)))


def permute_outcomes_stratified(
    batch: EpisodeBatch, n_buckets: int = 16, seed: int = 0
) -> EpisodeBatch:
    """shuffle settlements only among episodes the market priced alike.

    the plain permutation destroys the price's calibration, which is what turns
    the null world into an arbitrage. stratifying fixes that by construction: a
    contract trading near 0.20 only ever swaps outcomes with other contracts
    near 0.20, so each bucket keeps its own base rate and the price keeps
    predicting settlement exactly as well as it did.

    what gets removed is whatever distinguished episodes WITHIN a bucket, which
    is precisely "the information beyond the price". so this asks a narrower and
    arguably better question than the plain permutation: does the agent know
    anything the market does not?

    the bucket count trades two failures against each other. coarse buckets
    shuffle freely but let calibration drift back, recreating the arbitrage.
    fine buckets preserve calibration but become homogeneous, and shuffling
    inside a bucket where every outcome is identical changes nothing at all.
    whether a usable middle exists is an empirical question; see
    scripts/stratified_sweep.py.
    """
    rng = np.random.default_rng(seed)
    key = 0.5 * (np.asarray(batch.bid)[:, -1] + np.asarray(batch.ask)[:, -1])
    settle = np.asarray(batch.settlement).copy()

    # quantile edges, so every bucket holds a similar number of episodes
    # regardless of how the prices happen to pile up
    edges = np.quantile(key, np.linspace(0.0, 1.0, n_buckets + 1))
    # np.digitize on unique edges: ties collapse, which is correct here since a
    # degenerate bucket cannot be shuffled anyway
    bucket = np.digitize(key, np.unique(edges)[1:-1], right=False)

    out = settle.copy()
    for b in np.unique(bucket):
        idx = np.flatnonzero(bucket == b)
        out[idx] = settle[rng.permutation(idx)]

    return replace(
        batch, settlement=out, _market_features=batch.market_features()
    )


# ---------------------------------------------------------------------------
# measurements of what a relabelling does to the world. these live here rather
# than in a script because two scripts need them, and a scripts/__init__.py
# added solely to let one script import another turns a directory of entry
# points into a package for no reason.

def calibration(batch, n_bins: int = 8):
    """(bins, weighted mean absolute calibration error) at terminal mid."""
    mid = 0.5 * (batch.bid[:, -1] + batch.ask[:, -1])
    y = np.asarray(batch.settlement, dtype=float)
    edges = np.quantile(mid, np.linspace(0.0, 1.0, n_bins + 1))

    bins, err, total = [], 0.0, 0
    for i in range(n_bins):
        hi_inclusive = i == n_bins - 1
        m = (mid >= edges[i]) & ((mid <= edges[i + 1]) if hi_inclusive
                                 else (mid < edges[i + 1]))
        if m.sum() < 20:
            continue
        implied, realised, n = float(mid[m].mean()), float(y[m].mean()), int(m.sum())
        bins.append({"implied": implied, "realised": realised, "n": n})
        err += abs(implied - realised) * n
        total += n
    return bins, (err / total if total else float("nan"))


def fade_edge(batch, extreme: float = 0.1):
    """per-contract edge of fading the market's terminal price.

    the strategy needs nothing but the quote: sell when the market is confident
    the answer is yes, buy when it is confident the answer is no. in a
    calibrated market this earns nothing. after the outcomes are permuted the
    price still moves toward the TRUE outcome while the settlement belongs to a
    different contract, so the same rule collects the whole mispricing.

    this is the mechanism behind the permutation null's failure, and it is not
    a quirk of this market: wherever an observation is a forecast of the label,
    permuting the label makes that forecast exploitably wrong.
    """
    last = 0.5 * (batch.bid[:, -1] + batch.ask[:, -1])
    y = np.asarray(batch.settlement, dtype=float)
    edge = np.where(last > 0.5, last - y, y - last)
    ext = (last > 1.0 - extreme) | (last < extreme)
    return float(edge.mean()), float(edge[ext].mean()), int(ext.sum())

