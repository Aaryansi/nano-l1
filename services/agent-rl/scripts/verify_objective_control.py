"""Reconstruct new objective-control results from both episode endpoints.

This checks recorded computations and identities, without replaying training or
certifying feature fidelity. The selected research_v1 archive predates schema 2
and intentionally cannot pass this stronger, prospective evidence gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.features import MARKET_FEATURES
from nano_rl.explain.sanity import short_verdict, test_span_against_null
from portable_validation import reconstructed
from positive_control import direct_contrast
from run_manifest import digest

ARCHIVE = "positive_control_returns.npz"
TASKS = ["prediction", "trading"]
RETURN_KEYS = ("returns", "masked_returns", "reference_returns", "reference_masked_returns")
SEED_KEYS = ("agent_seeds", "observed_masking_seeds", "reference_policy_seeds",
             "reference_masking_seeds", "reference_blinding_seeds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(token):
        raise ValueError(f"non-finite objective-control JSON: {token}")
    return json.loads(path.read_text(), parse_constant=invalid)


def comparison_record(span, nulls):
    result = test_span_against_null(span, nulls)
    record = result.as_dict()
    if record["null_std"] <= 1e-12:
        record["z_score"] = record["p_normal"] = None
    return result, record


def verify_normalizer(split, recorded):
    """Permit float32 log1p dispatch roundoff only in the volume moments.

    The volume feature is evaluated before concatenation promotes it to
    float64. CPU math kernels can differ at float32 precision. A one-ULP
    pointwise allowance bounds the mean error and the population standard
    deviation error by the maximum allowance. Other features, metadata and
    all endpoint summaries retain the ordinary 1e-9 reconstruction gate.
    This is a validation tolerance, not a change to the fitted normalizer.
    """
    actual = split.normalizer.state_dict()
    volume = np.log1p(np.maximum(split.train.volume, 0.0))
    tolerance = 1e-9
    if volume.dtype == np.dtype("float32"):
        tolerance += float(np.max(np.spacing(volume)))
    index = MARKET_FEATURES.index("volume_rate")
    require(isinstance(recorded, dict) and recorded.keys() == actual.keys(),
            "objective normalizer keys differ")
    for key in actual:
        values = recorded[key]
        require(isinstance(values, list) and len(values) == len(actual[key])
                and all(type(value) is float and np.isfinite(value) for value in values),
                "objective normalizer shape, type or finiteness differs")
        # Do not hide a discontinuity in the near-zero variance guard.
        bound = tolerance if key == "mean" or actual["std"][index] != 1.0 else 1e-9
        require(abs(actual[key][index] - values[index]) <= bound,
                "objective volume normalizer differs beyond input precision")
        actual[key][index] = values[index]
    reconstructed(actual, recorded, "objective train-only normalizer")


def verify(reports, corpus):
    payload = read(reports / "positive_control.json")
    require(type(payload.get("endpoint_schema_version")) is int and payload["endpoint_schema_version"] == 2,
            "objective-control evidence requires schema 2 with both raw endpoints")
    require(payload["endpoint_returns_file"] == ARCHIVE, "objective archive name differs")
    require(digest(reports / ARCHIVE) == payload["endpoint_returns_sha256"], "objective archive hash differs")
    require(digest(corpus) == payload["corpus_sha256"], "objective corpus hash differs")
    split = walk_forward_split(EpisodeBatch.load(str(corpus)))
    require(type(payload["n_train"]) is int and payload["n_train"] == len(split.train)
            and type(payload["n_test"]) is int and payload["n_test"] == len(split.test),
            "objective split counts differ")
    verify_normalizer(split, payload["normalizer"])
    seed, n_null = payload["seed"], payload["n_null"]
    require(type(seed) is int and seed >= 0 and type(n_null) is int and n_null >= 2,
            "invalid objective seed or reference count")
    sweep = payload["budget_sweep"]
    require(isinstance(sweep, list) and len(sweep) > 0, "objective budget sweep is empty")
    budgets = [row["updates"] for row in sweep]
    require(all(type(n) is int and n > 0 for n in budgets) and len(set(budgets)) == len(budgets),
            "invalid objective budget membership")
    require(all([task["task"] for task in row["tasks"]] == TASKS for row in sweep),
            "objective task order differs")
    n_seeds = sweep[0]["tasks"][0]["n_seeds"]
    require(type(n_seeds) is int and n_seeds > 0, "invalid objective agent count")
    seeds = {"agent_seeds": np.arange(n_seeds) + seed,
             "observed_masking_seeds": np.full(n_seeds, seed),
             "reference_policy_seeds": np.arange(n_null) + 10000 + seed,
             "reference_masking_seeds": np.arange(n_null) + 10000 + seed,
             "reference_blinding_seeds": np.arange(n_null) + 3000}
    with np.load(reports / ARCHIVE, allow_pickle=False) as raw:
        require(set(raw.files) == set(RETURN_KEYS) | set(SEED_KEYS) | {"budgets", "task_names", "open_epoch"},
                "objective archive membership differs")
        for key, expected in {"budgets": np.asarray(budgets), "task_names": np.asarray(TASKS), **seeds}.items():
            require(raw[key].dtype.kind == expected.dtype.kind and np.array_equal(raw[key], expected),
                    f"objective {key} membership differs")
        identities = raw["open_epoch"]
        require(identities.shape == (len(split.test),) and np.isfinite(identities).all()
                and len(np.unique(identities)) == len(split.test)
                and np.array_equal(identities, split.test.open_epoch), "objective episode identities differ")
        for key in RETURN_KEYS:
            count = n_null if key.startswith("reference_") else n_seeds
            require(raw[key].shape == (len(budgets), 2, count, len(split.test))
                    and raw[key].dtype.kind == "f" and np.isfinite(raw[key]).all(),
                    f"objective endpoint shape or finiteness differs: {key}")
        for b, budget in enumerate(sweep):
            expected_tasks = []
            for t, task in enumerate(budget["tasks"]):
                nulls = (raw["reference_returns"][b, t].mean(axis=1)
                         - raw["reference_masked_returns"][b, t].mean(axis=1)).tolist()
                rows = []
                for s in range(n_seeds):
                    full, masked = raw["returns"][b, t, s], raw["masked_returns"][b, t, s]
                    span = float(full.mean() - masked.mean())
                    result, recorded = comparison_record(span, nulls)
                    rows.append({"seed": seed + s, "return": float(full.mean()), "span": span,
                                 "masking_seed": seed, "result": recorded, "fires": result.passes,
                                 "verdict": short_verdict(result)})
                primary = rows[0]
                expected = {"task": TASKS[t], "updates": budgets[b], "n_seeds": n_seeds,
                    **{key: primary[key] for key in ("return", "span", "result", "fires", "verdict")},
                    "per_seed": rows, "return_mean_across_seeds": float(np.mean([r["return"] for r in rows])),
                    "span_mean_across_seeds": float(np.mean([r["span"] for r in rows])),
                    "null_spans": nulls, "null_mean": float(np.mean(nulls)),
                    "null_std": float(np.std(nulls, ddof=1)),
                    **{key: seeds[key].tolist() for key in SEED_KEYS if key.startswith("reference_")},
                    "background_seed": seed, "background_samples": 192, "observation_moment_seed": seed}
                reconstructed(expected, task, f"objective budget {budgets[b]}/{TASKS[t]}")
                expected_tasks.append(expected)
            undecided = [task["task"] for task in expected_tasks if task["verdict"] == "unresolved"]
            expected_budget = {"updates": budgets[b], "tasks": expected_tasks,
                "contrast": direct_contrast(*expected_tasks), "undecided_tasks": undecided,
                "separated": None if undecided else expected_tasks[0]["fires"] and not expected_tasks[1]["fires"]}
            reconstructed(expected_budget, budget, f"objective budget {budgets[b]}")
    reconstructed(sweep[0], {key: payload[key] for key in sweep[0]}, "objective first-budget aliases")
    return {"budgets_checked": len(budgets), "observed_policies": len(budgets) * 2 * n_seeds,
            "reference_policies": len(budgets) * 2 * n_null, "episodes_per_endpoint": len(split.test),
            "scope": "both raw endpoints, recorded seed/corpus identities and computed summaries/decisions; no training or checkpoint replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.reports, args.corpus), indent=2))


if __name__ == "__main__":
    main()
