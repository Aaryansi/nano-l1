"""Reference-bank resampling diagnostics conditional on a fixed observed span.

The percentile interval uses only resamples with a defined scale. Report its
denominator and all three verdict-state counts separately. Decision retention
counts unresolved resamples as non-retention and is undefined if the original
reference cannot decide. These are descriptive sensitivity measures, not
confidence guarantees for a calibrated test or uncertainty in the observation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.explain.sanity import short_verdict, test_span_against_null


def bootstrap(observed: float, nulls: list[float], n_boot: int, seed: int) -> dict:
    if n_boot < 1:
        raise ValueError("bootstrap resamples must be positive")
    a = np.asarray(nulls, dtype=float)
    point = test_span_against_null(observed, a)
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(n_boot, len(a)))]
    mean, sd = draws.mean(axis=1), draws.std(axis=1, ddof=1)
    ok = sd > 1e-12
    z = np.full(n_boot, np.nan)
    z[ok] = (observed - mean[ok]) / sd[ok]
    finite = z[np.isfinite(z)]
    states = [short_verdict(test_span_against_null(observed, draw)) for draw in draws]
    counts = {label: states.count(label) for label in
              ("informative", "not distinguishable from null", "unresolved")}
    label = short_verdict(point)
    return {
        "z": float(point.z_score) if not point.degenerate_null and np.isfinite(point.z_score) else None,
        "z_lo": float(np.percentile(finite, 2.5)) if finite.size else None,
        "z_hi": float(np.percentile(finite, 97.5)) if finite.size else None,
        "verdict": label,
        "verdict_stability": counts[label] / n_boot if label != "unresolved" else None,
        "verdict_counts": counts,
        "n_defined_z": int(finite.size),
        "degenerate_fraction": float(1.0 - ok.mean()),
        "n_null": int(a.size), "n_boot": n_boot, "seed": seed,
        "scope": "reference-bank resampling conditional on fixed observed span; z interval conditional on finite-scale resamples; unresolved resamples do not retain a resolved decision",
    }


def collect(reports: Path) -> list[tuple[str, float, list[float]]]:
    """(label, observed, nulls) for every headline test in the paper."""
    out: list[tuple[str, float, list[float]]] = []

    def load(name):
        p = reports / name
        return json.loads(p.read_text()) if p.exists() else None

    # Settlement-permutation diagnostics.
    pm = load("permuted_null_test.json")
    if pm:
        for c in pm["cases"]:
            out.append((f"permuted: {c['case']}", c["span"], c["null_spans"]))

    # Matched-population noise-training diagnostics.
    m = load("matched_null_test.json")
    if m:
        for c in m["cases"]:
            out.append((f"matched: {c['case']}", c["span"], c["null_spans"]))

    # Synthetic reference-bank diagnostics.
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
    rows = collect(Path(args.reports))
    if not rows:
        raise SystemExit(f"no result artifacts found in {args.reports}")
    print("reference-bank resampling; observed spans held fixed")
    results = {}
    for label, observed, nulls in rows:
        r = bootstrap(observed, nulls, args.n_boot, args.seed)
        results[label] = r
        point = "undefined" if r["z"] is None else f"{r['z']:+.2f}"
        interval = ("undefined" if r["z_lo"] is None else
                    f"[{r['z_lo']:+.2f}, {r['z_hi']:+.2f}]")
        retention = "unresolved" if r["verdict_stability"] is None else f"{r['verdict_stability']:.1%}"
        print(f"  {label}: z {point}, conditional interval {interval}; "
              f"defined z {r['n_defined_z']}/{args.n_boot}; decision retention {retention}")
    print("  All three verdict counts are retained; this does not measure uncertainty")
    print("  in observed returns or establish calibration or equivalence.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "z_intervals.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
