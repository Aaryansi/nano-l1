"""Describe within-price-bucket settlement permutations across resolutions.

Label-change fractions, label correlations, calibration error and static gross
fade payoffs describe this finite corpus. They do not measure conditional
information removal or establish randomization validity. The fixed thresholds
are exploratory screening heuristics. Failure to meet them is not proof that a
conditional null is impossible or vacuous. A conditional null may legitimately
preserve predictive information in its conditioning price.
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
    # Descriptive marginal correlation; this does not measure conditional
    # information removal or establish a valid sequential randomization law.
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
    print(f"  exploratory screen: changed >= {args.min_changed:.0%}, "
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
        print("  No tested bucket count meets these exploratory thresholds.")
    print("  Label correlation and change fractions are not conditional-information")
    print("  estimates. This screen does not prove a conditional null valid or impossible.")

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
        "screen_scope": "exploratory thresholds, not conditional validity or information removal",
        "usable_exists": bool(usable),
        "best": (max(usable, key=lambda r: r["changed"]) if usable else None),
    }, indent=2))
    print(f"\nwrote {out}/stratified_sweep.json")


if __name__ == "__main__":
    main()
