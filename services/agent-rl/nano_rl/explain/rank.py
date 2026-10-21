"""Standard finite ranks under an explicit joint-exchangeability assumption.

The observed and reference statistics must be jointly exchangeable. The rank
is super-uniform marginally over that joint draw; it need not be calibrated
conditional on a fixed reference bank. See Kuchibhotla (2020), arXiv:2005.06095.
This module cannot establish exchangeability for a chosen scientific reference.
"""
from math import fsum

import numpy as np


def exchangeable_rank_p(stat, reference, *, alternative="greater") -> float:
    """Include the observation and ties; use a symmetric pooled two-sided score."""
    refs = np.asarray(reference, dtype=float)
    if refs.ndim != 1 or not len(refs) or not np.isfinite(refs).all() or not np.isfinite(stat):
        raise ValueError("rank comparisons require a finite observation and nonempty finite reference vector")
    pool = np.append(refs, float(stat))
    if alternative == "greater":
        scores = pool
    elif alternative == "two-sided":
        # Scaling prevents overflow; sorting makes floating-point centering
        # invariant to which point was designated as the observation.
        scaled = pool / max(1., float(np.max(np.abs(pool))))
        center = fsum(sorted(scaled)) / len(pool)
        scores = np.abs(scaled - center)
    else:
        raise ValueError("alternative must be greater or two-sided")
    return float(np.count_nonzero(scores >= scores[-1]) / len(pool))


def rank_comparisons(stat, reference, *, alpha=.05) -> dict:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    floor = 1. / (len(reference) + 1)
    return {alternative: {"p_value": p, "rejected": p <= alpha, "alpha": alpha,
                          "rank_floor": floor, "rank_floor_at_or_below_alpha": floor <= alpha}
            for alternative in ("greater", "two-sided")
            for p in [exchangeable_rank_p(stat, reference, alternative=alternative)]}
