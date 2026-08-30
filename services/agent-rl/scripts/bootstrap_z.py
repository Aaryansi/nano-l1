"""confidence intervals on the headline z-scores.

every z reported in this paper has the form

    z = (observed - mean(nulls)) / sd(nulls)

and both the mean and the sd are estimates from a finite number of null agents,
typically 12 to 24. reporting z to two decimals hides that. a reviewer is
entitled to ask how much of "z = +0.23" is signal and how much is the accident
of which twenty-four agents we happened to train.

this resamples the null draws with replacement and recomputes z on each
replicate, giving a percentile interval for z and, more usefully, the fraction
of replicates in which the VERDICT is unchanged. an interval that straddles the
decision boundary is a result that should not be reported as decided.

what this does NOT capture: uncertainty in the observed statistic itself, which
is a mean over held-out episodes. the null sd is the dominant term and the one
the small sample makes fragile, so it is the one bootstrapped here. that
limitation is stated in the paper rather than papered over.

usage:
    python scripts/bootstrap_z.py --out ../../reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.explain.sanity import test_span_against_null  # noqa: E402


def bootstrap(observed: float, nulls: list[float], n_boot: int, seed: int) -> dict:
    """percentile interval for z, and how often the VERDICT survives.

    stability is measured under the paper's actual decision rule, which
    requires the rank and normal criteria to reject together, by calling the
    same function the results use. an earlier version of this script tested
    |z| > 1.96 alone and reported one headline as unstable that the real
    two-part rule never called significant in the first place. a robustness
    check that does not replicate the decision it is checking measures nothing.
    """
    a = np.asarray(nulls, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    draws = a[idx]

    mean = draws.mean(axis=1)
    sd = draws.std(axis=1, ddof=1)

    # a degenerate resample has no scale, so z is not merely large but
    # undefined. those replicates are counted separately rather than being
    # silently dropped or clamped, either of which would bias the interval.
    ok = sd > 1e-12
    z = np.full(n_boot, np.nan)
    z[ok] = (observed - mean[ok]) / sd[ok]

    point = test_span_against_null(observed, a)
    finite = z[np.isfinite(z)]

    if finite.size == 0:
        return {"z": point.z_score, "degenerate_fraction": 1.0, "n_boot": n_boot}

    passes = np.fromiter(
        (test_span_against_null(observed, draws[i]).passes
         for i in np.flatnonzero(ok)),
        dtype=bool, count=int(ok.sum()),
    )
    agree = float(np.mean(passes == bool(point.passes)))
    return {
        "z": float(point.z_score),
        "z_lo": float(np.percentile(finite, 2.5)),
        "z_hi": float(np.percentile(finite, 97.5)),
        "verdict": "informative" if point.passes else "not distinguishable",
        "verdict_stability": agree,
        "degenerate_fraction": float(1.0 - ok.mean()),
        "n_null": int(a.size),
        "n_boot": n_boot,
    }


def collect(reports: Path) -> list[tuple[str, float, list[float]]]:
    """(label, observed, nulls) for every headline test in the paper."""
    out: list[tuple[str, float, list[float]]] = []

    def load(name):
        p = reports / name
        return json.loads(p.read_text()) if p.exists() else None

    # the construction the paper adopts: outcomes permuted, agents sighted,
    # corpus fixed. listed first because these are the headline verdicts.
    pm = load("permuted_null_test.json")
    if pm:
        for c in pm["cases"]:
            out.append((f"permuted: {c['case']}", c["span"], c["null_spans"]))

    # the corpus-matched construction, reported as rejected
    m = load("matched_null_test.json")
    if m:
        for c in m["cases"]:
            out.append((f"matched: {c['case']}", c["span"], c["null_spans"]))

    # the earlier construction, kept because the paper reports it as what was
    # tried before rather than dropping it silently
    s = load("sanity_test.json")
    if s:
        for key, label in (("planted_signal", "planted signal"),
                           ("held_out_null", "null corpus"),
                           ("real_market", "real market")):
            out.append((f"shared-null: {label}",
                        s["results"][key]["statistic"], s["null_spans"]))

    m = load("manifold_masking.json")
    if m:
        n = m["conditional_null_spans"]
        out.append(("planted, conditional masking",
                    m["conditional_planted_signal"]["statistic"], n))
        out.append(("real market, conditional masking",
                    m["conditional_real_market"]["statistic"], n))

    p = load("positive_control.json")
    if p:
        for t in p["tasks"]:
            out.append((f"real {t['task']}", t["span"], t["null_spans"]))

    # the blinded-real null at several training budgets. the width shrinks as
    # its agents converge, so the margin is reported per budget rather than as
    # a single number.
    b = load("null_budget_check.json")
    if b:
        for row in b["rows"]:
            out.append((f"blinded-real, {row['updates']} updates",
                        b["observed_span"], row["spans"]))

    # the same agent against two null constructions. this pair decides a
    # headline, so its interval matters more than any other here.
    c = load("null_corpus_check.json")
    if c:
        for key, label in (("null_synthetic", "market vs synthetic-corpus null"),
                           ("null_blinded_real", "market vs blinded-real null")):
            out.append((label, c["observed_span"], c[key]["spans"]))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="../../reports")
    ap.add_argument("--out", default="../../reports")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reports = Path(args.reports)
    rows = collect(reports)
    if not rows:
        raise SystemExit(f"no result artifacts found in {reports}")

    print("=" * 84)
    print("bootstrap confidence intervals on the reported z-scores")
    print("=" * 84)
    print(f"  {args.n_boot} resamples of the null draws, percentile interval.")
    print("  stability is measured under the paper's own two-part rule.\n")
    print(f"  {'test':<34}{'n':>4}{'z':>9}{'95% interval':>20}{'verdict held':>14}")

    results = {}
    for label, observed, nulls in rows:
        r = bootstrap(observed, nulls, args.n_boot, args.seed)
        results[label] = r
        if "z_lo" not in r:
            print(f"  {label:<34}{'--':>4}{r['z']:>9.2f}{'degenerate':>20}"
                  f"{'--':>14}")
            continue
        interval = f"[{r['z_lo']:+.2f}, {r['z_hi']:+.2f}]"
        print(f"  {label:<34}{r['n_null']:>4}{r['z']:>+9.2f}{interval:>20}"
              f"{r['verdict_stability']:>13.0%}")

    weakest = min(
        (r for r in results.values() if "verdict_stability" in r),
        key=lambda r: r["verdict_stability"],
    )
    print(f"\n  the least stable verdict holds in "
          f"{weakest['verdict_stability']:.0%} of resamples")
    if weakest["verdict_stability"] < 0.95:
        print("  at least one verdict is not robust to the null sample and the")
        print("  paper must report it as undecided rather than as decided.")
    else:
        print("  every reported verdict survives resampling of its own null.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "z_intervals.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/z_intervals.json")


if __name__ == "__main__":
    main()
