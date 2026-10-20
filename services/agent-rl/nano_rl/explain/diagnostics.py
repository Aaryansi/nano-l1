"""Descriptive attribution diagnostics with explicit treatment of ties."""
from __future__ import annotations

import numpy as np


def magnitude_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of magnitudes; constant rankings are undefined."""
    def ranks(values):
        _, inverse, counts = np.unique(np.abs(values), return_inverse=True,
                                       return_counts=True)
        averages = np.cumsum(counts) - (counts - 1) / 2
        return averages[inverse]

    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("rank comparison requires finite vectors of equal shape")
    ra, rb = ranks(a), ranks(b)
    if len(a) < 2 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
