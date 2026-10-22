"""Describe calibration and static gross payoffs after settlement shuffling.

Preserving feature arrays and the marginal settlement rate does not preserve
the decision problem. Independent shuffling can make buying below the marginal
settlement rate and selling above it profitable in a frictionless one-period
model. Terminal-price diagnostics here are empirical gross payoffs, without
execution costs or a live trading claim. They do not establish efficiency of
the original market or invalidate permutation tests for other hypotheses.
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
from nano_rl.env.permuted import (  # noqa: E402
    calibration,
    fade_edge,
    outcome_rate,
    permute_outcomes,
)




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
    print("  Shuffling changes price-settlement dependence and the reward law.")
    print("  Price remains reward-relevant through the acquisition cost.")

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
    print(f"  Static gross payoff at 100 contracts on permuted extreme-price episodes: "
          f"${100 * fades['permuted']['extremes']:.2f} per selected episode.")
    print("  These terminal-price diagnostics omit costs and execution constraints.")
    print("  They are not evidence of live trading profit or market efficiency.")

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
