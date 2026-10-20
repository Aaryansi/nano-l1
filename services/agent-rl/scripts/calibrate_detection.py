"""Estimate detection rates on independent synthetic trials conditional on a reference bank."""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.env.synthetic import synthetic_train_eval, make_learnable_corpus
from nano_rl.env.features import fit_normalizer
from nano_rl.explain.rollout import VectorizedRollout, greedy_policy
from nano_rl.explain.sanity import check_resolving_power, test_span_against_null, UNRESOLVED
from sanity_check_explanations import build_null_distribution, train, endpoint_span


def wilson_interval(successes: int, trials: int, z=1.959963984540054):
    """Binomial interval for independent trials, conditional on the fixed bank."""
    if trials < 1:
        return None
    p = successes / trials
    scale = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / scale
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / scale
    return [max(0., center - half), min(1., center + half)]


def summarize(rows):
    resolved = [row for row in rows if row["result"]["verdict"] != UNRESOLVED]
    rejected = sum(row["result"]["passes"] for row in resolved)
    positive = sum(row["result"]["passes"] and row["result"]["z_score"] > 0 for row in resolved)
    return {"n_trials": len(rows), "n_resolved": len(resolved),
            "n_unresolved": len(rows) - len(resolved), "n_rejected": rejected,
            "n_positive_rejections": positive,
            "rejection_rate_resolved": rejected / len(resolved) if resolved else None,
            "rejection_rate_ci95": wilson_interval(rejected, len(resolved)),
            "positive_detection_rate_resolved": positive / len(resolved) if resolved else None,
            "positive_detection_ci95": wilson_interval(positive, len(resolved))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-null", type=int, default=24)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0., .2, 1.])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-json", type=Path)
    args = ap.parse_args()
    if (args.trials < 1 or args.updates < 1 or len(set(args.strengths)) != len(args.strengths)
            or any(not np.isfinite(s) or s < 0 for s in args.strengths)):
        raise SystemExit("trials/updates must be positive and strengths must be distinct")
    check_resolving_power(args.n_null, what="the calibration reference")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reference_json:
        reference = json.loads(args.reference_json.read_text())
        if reference.get("training_updates") != args.updates or len(reference["null_spans"]) != args.n_null:
            raise SystemExit("reference bank must match the requested training and draw budgets")
        nulls = reference["null_spans"]
    else:
        nulls = build_null_distribution(args.n_null, args.updates, 0, 0, args.seed + 100000)
    cases, raw = [], {}
    for strength_index, strength in enumerate(args.strengths):
        rows = []
        for trial in range(args.trials):
            corpus_seed = 500000 + args.seed + trial
            policy_seed = 800000 + args.seed + strength_index * args.trials + trial
            pair = synthetic_train_eval(make_learnable_corpus, n_episodes=1200,
                                        signal_strength=strength, seed=corpus_seed)
            norm = fit_normalizer(pair.train)
            agent = train(pair.train, norm, args.updates, policy_seed)
            span = endpoint_span(agent, pair.evaluation, norm, policy_seed)
            result = test_span_against_null(span, nulls)
            returns = VectorizedRollout(pair.evaluation, normalizer=norm).run(greedy_policy(agent))["returns"]
            raw[f"strength_{strength_index}_trial_{trial}"] = returns
            rows.append({"trial": trial, "policy_seed": policy_seed, "data": pair.metadata(),
                         "return": float(returns.mean()), "result": result.as_dict()})
            print(f"strength {strength:g}, trial {trial + 1}/{args.trials}: {result.summary()}", flush=True)
        cases.append({"signal_strength": strength, **summarize(rows), "trials": rows})
    payload = {"design": "independent observed trials conditional on one independently trained reference bank",
               "interval_scope": "pointwise conditional binomial uncertainty; excludes variation across reference banks",
               "updates": args.updates, "reference_json": str(args.reference_json) if args.reference_json else None,
               "reference_base_seed": args.seed + 100000 if not args.reference_json else None,
               "null_spans": nulls, "cases": cases}
    (args.out / "detection_calibration.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(args.out / "detection_calibration_returns.npz", **raw)


if __name__ == "__main__":
    main()
