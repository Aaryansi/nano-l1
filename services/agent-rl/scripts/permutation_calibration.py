"""what the outcome permutation actually removes, measured.

the permutation null claims to remove the information and nothing else. the
first half of that is checked by tests: every array except `settlement` is bit
identical, so no observation moves. the second half is an empirical question
about what the information WAS, and it has an uncomfortable answer worth
reporting rather than leaving for a reviewer to find.

the market's price is well calibrated: a contract trading at 0.83 settles at 1
about 83% of the time. permuting the outcomes across episodes destroys that.
every price bucket then settles near the base rate, which is exactly the
information removal the null needs, but it also means the null environment is a
market whose prices are WRONG. buying at 0.90 wins 90% of the time in the real
world and about 50% of the time in the null, so trading is strictly more
punishing there than in the environment the observed agent faced.

this measures the size of that asymmetry so the paper can state it. a stratified
permutation, shuffling outcomes only within price buckets, would preserve
calibration and remove only the information beyond the price; that is a
different and narrower null, and it is named as future work rather than run.

the calibration numbers here use terminal mid-price on the training split with
eight quantile bins. that is not the estimator behind the figure quoted in the
setup section, so the two are not directly comparable and only the contrast
between the real and permuted columns is meant to be read.

usage:
    python scripts/permutation_calibration.py --corpus data/corpus/corpus_candles_60s_spot.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import EpisodeBatch  # noqa: E402
from nano_rl.env.permuted import outcome_rate, permute_outcomes  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()

    split = walk_forward_split(EpisodeBatch.load(args.corpus))
    real = split.train
    perm = permute_outcomes(real, seed=args.seed)

    print("=" * 78)
    print("what the outcome permutation removes")
    print("=" * 78)
    print(f"  outcome rate  real {outcome_rate(real):.4f}   "
          f"permuted {outcome_rate(perm):.4f}   (a permutation preserves it)\n")

    res = {}
    print(f"  {'implied':>9}{'realised (real)':>18}{'realised (permuted)':>22}"
          f"{'n':>8}")
    rb, rerr = calibration(real)
    pb, perr = calibration(perm)
    for r, p in zip(rb, pb):
        print(f"  {r['implied']:>9.3f}{r['realised']:>18.3f}"
              f"{p['realised']:>22.3f}{r['n']:>8}")
    print(f"\n  weighted mean absolute calibration error")
    print(f"    real     {rerr:.4f}")
    print(f"    permuted {perr:.4f}")
    print()
    print("  the price stops predicting the outcome, which is the information")
    print("  the null is meant to remove. it also means the null environment")
    print("  prices contracts wrongly, so trading there is strictly more")
    print("  punishing than in the environment the observed agent faced.")

    print()
    print("  per-contract edge of fading the terminal price, a rule needing")
    print("  nothing but the quote:")
    print(f"  {'':>11}{'all episodes':>16}{'extremes only':>17}{'n':>8}")
    fades = {}
    for label, b in (("real", real), ("permuted", perm)):
        a, e, n = fade_edge(b)
        fades[label] = {"all": a, "extremes": e, "n_extreme": n}
        print(f"  {label:>11}{a:>16.4f}{e:>17.4f}{n:>8}")
    print()
    print(f"  at the 100-contract position cap that is about "
          f"${100 * fades['permuted']['extremes']:.0f} an episode in the")
    print("  permuted world and nothing in the real one. the null agents do not")
    print("  have LESS to learn than the observed agent. they have far more.")

    res = {"fade_edge": fades,
           "outcome_rate_real": outcome_rate(real),
           "outcome_rate_permuted": outcome_rate(perm),
           "calibration_error_real": rerr,
           "calibration_error_permuted": perr,
           "bins_real": rb, "bins_permuted": pb}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_calibration.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/permutation_calibration.json")


if __name__ == "__main__":
    main()
