"""is there a bucket width that removes information and keeps calibration?

the plain outcome permutation fails because it destroys the price's calibration
and hands the null agents an arbitrage. stratifying the permutation by price
fixes that by construction, but introduces a tension that may have no solution:

  coarse buckets   plenty of shuffling, but the bucket spans a wide price range
                   so calibration drifts back and the arbitrage returns.
  fine buckets     calibration preserved, but each bucket becomes homogeneous,
                   and shuffling inside a bucket where every outcome is already
                   identical removes nothing.

this sweeps the bucket count and measures both sides, without training anything,
so the question is answered in seconds rather than hours. only if a usable
window exists is it worth running the full null test there.

the two quantities to watch:

  removal      fraction of episodes whose settlement changed, and how far the
               permuted labels decorrelate from the originals. this is how much
               information the construction actually destroys.
  distortion   calibration error and the per-contract edge of fading the
               terminal price. this is how much the construction breaks the
               world, which is what killed the unstratified version.

usage:
    python scripts/stratified_sweep.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
    permute_outcomes,
    permute_outcomes_stratified,
)


def measure(real, perm) -> dict:
    """how much was removed, and how much was broken."""
    a = np.asarray(real.settlement, dtype=float)
    b = np.asarray(perm.settlement, dtype=float)

    changed = float((a != b).mean())
    # correlation between the original and permuted labels. 1.0 means nothing
    # was removed; 0.0 means the label is now independent of the truth.
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0

    _, cal = calibration(perm)
    _, fade_ext, _ = fade_edge(perm)
    return {"changed": changed, "label_corr": corr,
            "calibration_error": cal, "fade_edge": fade_ext}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--buckets", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--seed", type=int, default=0)
    # a construction is usable only if it removes enough to be a null and
    # distorts little enough not to be a different world
    ap.add_argument("--max-cal-error", type=float, default=0.05)
    ap.add_argument("--max-fade-edge", type=float, default=0.05)
    ap.add_argument("--min-changed", type=float, default=0.10)
    args = ap.parse_args()

    real = walk_forward_split(EpisodeBatch.load(args.corpus)).train
    _, cal_real = calibration(real)
    _, fade_real, _ = fade_edge(real)

    # the diagnostic that explains whatever the sweep finds. if the price
    # already determines the outcome, then "information beyond the price" is a
    # small quantity and a null that removes only it cannot differ much from
    # the real data, at any bucket width.
    last = 0.5 * (np.asarray(real.bid)[:, -1] + np.asarray(real.ask)[:, -1])
    y = np.asarray(real.settlement, dtype=float)
    ext = (last > 0.9) | (last < 0.1)
    resolved = float(ext.mean())
    agree = float(np.mean((last[ext] > 0.5) == (y[ext] > 0.5)))
    price_r = float(np.corrcoef(last, y)[0, 1])

    print("=" * 82)
    print("stratified permutation: does a usable bucket width exist?")
    print("=" * 82)
    print(f"  the real corpus: calibration error {cal_real:.4f}, "
          f"fade edge {fade_real:+.4f}")
    print(f"  usable means: changed >= {args.min_changed:.0%}, "
          f"calibration error <= {args.max_cal_error}, "
          f"fade edge <= {args.max_fade_edge}")
    print(f"  the price alone: r = {price_r:.3f} with the outcome, and "
          f"{resolved:.1%} of episodes")
    print(f"  end within 0.1 of a bound, where it is right {agree:.1%} of the "
          f"time\n")

    rows = []
    print(f"  {'buckets':>8}{'changed':>10}{'label corr':>12}"
          f"{'calib err':>11}{'fade edge':>11}{'usable':>9}")

    for n in args.buckets:
        m = measure(real, permute_outcomes_stratified(real, n, args.seed))
        m["n_buckets"] = n
        m["usable"] = bool(
            m["changed"] >= args.min_changed
            and m["calibration_error"] <= args.max_cal_error
            and abs(m["fade_edge"]) <= args.max_fade_edge
        )
        rows.append(m)
        print(f"  {n:>8}{m['changed']:>10.3f}{m['label_corr']:>12.3f}"
              f"{m['calibration_error']:>11.4f}{m['fade_edge']:>+11.4f}"
              f"{('yes' if m['usable'] else 'no'):>9}")

    # the unstratified version is bucket count 1, and is the thing that failed
    m = measure(real, permute_outcomes(real, args.seed))
    m.update(n_buckets=1, usable=False)
    rows.insert(0, m)
    print(f"  {1:>8}{m['changed']:>10.3f}{m['label_corr']:>12.3f}"
          f"{m['calibration_error']:>11.4f}{m['fade_edge']:>+11.4f}"
          f"{'no':>9}   (unstratified)")

    usable = [r for r in rows if r["usable"]]
    print()
    print("=" * 82)
    if usable:
        best = max(usable, key=lambda r: r["changed"])
        print(f"  a usable window EXISTS. {len(usable)} bucket counts qualify;")
        print(f"  the most aggressive is {best['n_buckets']} buckets, which "
              f"changes {best['changed']:.1%} of")
        print(f"  outcomes at calibration error {best['calibration_error']:.4f} "
              f"and fade edge {best['fade_edge']:+.4f}.")
        print("  worth running the full null test there.")
    else:
        print("  NO usable window. every bucket count either fails to remove")
        print("  enough information or distorts the world enough to matter.")
        print("  the tension is not resolvable by tuning the bucket width, and")
        print("  the diagnostic above says why. the terminal price already")
        print(f"  correlates {price_r:.3f} with the outcome, so 'information beyond")
        print("  the price' lives in a small minority of episodes. shuffling")
        print("  within price buckets therefore leaves most labels untouched at")
        print("  ANY resolution. the construction is sound and vacuous here:")
        print("  it removes the right thing, and in this market there is almost")
        print("  none of it to remove.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stratified_sweep.json").write_text(json.dumps({
        "real": {"calibration_error": cal_real, "fade_edge": fade_real,
                 "price_outcome_r": price_r, "fraction_resolved": resolved,
                 "price_agrees_when_resolved": agree},
        "rows": rows,
        "thresholds": {"max_cal_error": args.max_cal_error,
                       "max_fade_edge": args.max_fade_edge,
                       "min_changed": args.min_changed},
        "usable_exists": bool(usable),
        "best": (max(usable, key=lambda r: r["changed"]) if usable else None),
    }, indent=2))
    print(f"\nwrote {out}/stratified_sweep.json")


if __name__ == "__main__":
    main()
