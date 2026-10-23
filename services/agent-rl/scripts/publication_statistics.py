"""Derive manuscript statistics without rewriting immutable experiment inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify_release import digest, require, verify
from nano_rl.explain.sanity import short_verdict
from nano_rl.metrics import cluster_bootstrap_ci, cluster_bootstrap_p_value


def read(path):
    return json.loads(path.read_text())


def close(actual, recorded, context):
    require(np.asarray(actual).shape == np.asarray(recorded).shape
            and np.allclose(actual, recorded, rtol=0, atol=1e-9),
            f"publication statistic differs from raw returns: {context}")


def nullable_z(result):
    value = result["z_score"]
    return float(value) if (result["null_std"] > 1e-12 and value is not None
                            and np.isfinite(value)) else None


def comparison(result):
    return {"span": result["statistic"], "reference_mean": result["null_mean"],
            "reference_std": result["null_std"], "standardized_distance": nullable_z(result),
            "legacy_tail_score": result["p_rank"],
            "legacy_normal_tail": result["p_normal"] if result["null_std"] > 1e-12 else None,
            "legacy_verdict": short_verdict(result), "reference_draws": result["n_null_samples"]}


def seed_summary(returns):
    values = np.asarray(returns, dtype=float)
    require(values.ndim == 2 and min(values.shape) > 0 and np.isfinite(values).all(),
            "require finite seed-by-episode returns")
    means = values.mean(axis=1)
    return {"mean": float(values.mean()), "seed_means": means.tolist(),
            "seed_sd": float(means.std(ddof=1)) if len(means) > 1 else None,
            "seed_sd_ddof": 1, "n_seeds": len(means), "n_episodes": values.shape[1]}


def state_statistics(reports, *, market_returns=None, market_epochs=None):
    path = reports / "state_masking.json"
    if not path.exists():
        return None
    payload, cases = read(path), []
    with np.load(reports / "state_masking_returns.npz", allow_pickle=False) as raw:
        for case in payload["cases"]:
            name = case["case"]
            original = raw[f"{name}__unmodified"]
            summary = seed_summary(original)
            identities = raw[f"{name}__open_epoch"]
            require(identities.shape == (case["n_episodes"],)
                    and summary["n_episodes"] == case["n_episodes"]
                    and summary["n_seeds"] == len(case["per_seed"])
                    and np.isfinite(identities).all() and len(np.unique(identities)) == len(identities),
                    "state-intervention membership differs")
            close(original.mean(axis=1), [row["unmodified_return"] for row in case["per_seed"]], name)
            if name == "market" and market_returns is not None:
                require(original.shape == market_returns.shape and np.array_equal(identities, market_epochs),
                        "state-intervention and primary evaluation membership differ")
                close(original, market_returns, "state-intervention unmodified market returns")
            blocks = (np.floor(identities / 86400).astype(int) if name == "market"
                      else np.arange(len(identities)))
            modes = {}
            for mode, recorded in case["interventions"].items():
                masked = raw[f"{name}__{mode}"]
                require(masked.shape == original.shape and np.isfinite(masked).all(),
                        "state-intervention endpoint shape differs")
                difference = original - masked
                interval = cluster_bootstrap_ci(list(original), list(masked), block_ids=blocks)
                close(difference.mean(), recorded["mean_span"], f"{name}/{mode} mean")
                close(difference.mean(axis=1), recorded["per_seed_spans"], f"{name}/{mode} seeds")
                close(interval, recorded["ci95"], f"{name}/{mode} interval")
                close(difference.mean(axis=1), [row["spans"][mode] for row in case["per_seed"]],
                      f"{name}/{mode} individual spans")
                modes[mode] = {"mean_span": float(difference.mean()),
                               "per_seed_spans": difference.mean(axis=1).tolist(),
                               "ci95": list(map(float, interval))}
            cases.append({"case": name, **summary, "interventions": modes,
                          "n_resampling_blocks": len(np.unique(blocks)),
                          "block_unit": "UTC day" if name == "market" else "episode"})
    return {"cases": cases, "scope": payload["limitations"]}


def steering_statistics(reports):
    path = reports / "steering.json"
    if not path.exists():
        return None
    payload, populations = read(path), {}
    for name in ("real_market", "learnable_synthetic"):
        rows = payload[name]
        require(rows and rows[0]["coef"] == 0, "steering baseline must be the declared zero penalty")
        baseline = np.asarray(rows[0]["pnl"], dtype=float)
        identity_key = "market" if name == "real_market" else "synthetic"
        identities = np.asarray(payload["evaluation_open_epoch"][identity_key], dtype=float)
        require(identities.shape == baseline.shape[1:] and np.isfinite(identities).all()
                and len(np.unique(identities)) == len(identities), "steering episode membership differs")
        blocks = (np.floor(identities / 86400).astype(int) if name == "real_market"
                  else np.arange(len(identities)))
        result = []
        for row in rows:
            values = np.asarray(row["pnl"], dtype=float)
            summary = seed_summary(values)
            require(values.shape == baseline.shape, "steering paired endpoint shape differs")
            close(summary["mean"], row["return_mean"], f"{name} return mean")
            close(summary["seed_means"], row["return_per_seed"], f"{name} seed returns")
            shares = np.asarray(row["target_share_per_seed"], dtype=float)
            require(shares.shape == (len(values),) and np.isfinite(shares).all()
                    and np.all((shares >= 0) & (shares <= 1)), "invalid steering attribution shares")
            close(shares.mean(), row["target_share_mean"], f"{name} attribution mean")
            data = {"penalty": row["coef"], **summary,
                    "target_share_mean": float(shares.mean()), "target_shares": shares.tolist()}
            if row["coef"] != 0:
                interval = cluster_bootstrap_ci(list(values), list(baseline), block_ids=blocks)
                probability = cluster_bootstrap_p_value(list(values), list(baseline), block_ids=blocks)
                difference = float((values - baseline).mean())
                close(interval, row["return_diff_ci"], f"{name} paired interval")
                close(probability, row["p_cluster"], f"{name} paired probability")
                close(difference, row["return_diff"], f"{name} paired mean")
                data.update(return_difference=difference, return_diff_ci95=list(map(float, interval)),
                            paired_bootstrap_probability=float(probability))
            result.append(data)
        populations[name] = result
        require(any(row["penalty"] == 20. for row in result), "declared steering penalty missing")
    return {"populations": populations, "resampling": payload["resampling"],
            "reported_penalty": 20., "scope": "all settings retained; no equivalence or noninferiority claim"}


def objective_statistics(reports):
    path = reports / "positive_control.json"
    if not path.exists():
        return None
    payload, result = read(path), []
    with np.load(reports / "positive_control_returns.npz", allow_pickle=False) as raw:
        require(raw["budgets"].tolist() == [row["updates"] for row in payload["budget_sweep"]],
                "objective budget membership differs")
        values = raw["returns"]
        require(values.ndim == 4 and values.shape[:2] == (len(payload["budget_sweep"]), 2)
                and values.shape[-1] == len(raw["open_epoch"])
                and np.isfinite(values).all(), "invalid objective-control raw returns")
        for budget_index, budget in enumerate(payload["budget_sweep"]):
            tasks = []
            for task_index, task in enumerate(budget["tasks"]):
                summary = seed_summary(values[budget_index, task_index])
                require(summary["n_seeds"] == task["n_seeds"] == len(task["per_seed"]),
                        "objective seed count differs")
                close(summary["seed_means"], [row["return"] for row in task["per_seed"]],
                      f"objective {task['task']}")
                tasks.append({"task": task["task"], **summary,
                              "recorded_span_comparisons": [comparison(row["result"]) for row in task["per_seed"]]})
            result.append({"updates": budget["updates"], "tasks": tasks})
    return {"budget_sweep": result,
            "scope": "returns reconstructed from episodes; secondary masked spans retained as recorded means; different reward units are not directly comparable"}


def derive(reports):
    results = read(reports / "results.json")
    with np.load(reports / "evaluation_returns.npz", allow_pickle=False) as raw:
        market_returns, market_epochs = raw["ppo_pnl"], raw["open_epoch"]
        market = {"costs": seed_summary(raw["ppo_pnl"]),
                  "zero_cost": seed_summary(raw["ppo_nofric_pnl"]),
                  "primary_uncertainty": results["ppo_seed_day_bootstrap"],
                  "baseline_mean_returns": {name: float(raw[f"baseline_{name}_pnl"].mean())
                                            for name in results["baselines"]}}
    headline_path = reports / "sanity_test.json"
    if not headline_path.exists():
        headline_path = reports / "headline/sanity_test.json"
    headline = read(headline_path)
    primary = {"cases": {name: comparison(row) for name, row in headline["results"].items()},
               "market_per_seed": [comparison(row) for row in headline["real_results_per_seed"]],
               "primary_checkpoint": headline["primary_real_checkpoint"],
               "scope": "synthetic bank; market comparisons descriptive; legacy rule lacks general calibration"}
    calibration = read(reports / "detection_calibration.json")
    rates = {"cases": [{key: value for key, value in case.items() if key != "trials"}
                       for case in calibration["cases"]],
             "scope": calibration["interval_scope"], "rank_validity_scope": calibration["rank_validity_scope"]}
    controls = {}
    for directory in (reports, reports / "seeds", reports / "resample", reports / "controls"):
        path = directory / "generalize_gym.json"
        if not path.exists():
            continue
        tasks = []
        for task in read(path):
            rows = task["checkpoints"]
            by_reference = {}
            for reference in ("env", "weight"):
                labels = [row[f"verdict_{reference}"] for row in rows]
                by_reference[reference] = {"outside": labels.count("informative"),
                                           "unresolved": labels.count("unresolved"),
                                           "within": labels.count("not distinguishable from null")}
                require(sum(by_reference[reference].values()) == len(labels),
                        "unknown control reference verdict")
            tasks.append({"task": task["env_id"], "unit": task["unit"],
                          "n_units": len(rows), "eligible": task["n_eligible_agents"],
                          "mean_return": float(np.mean([row["return"] for row in rows])),
                          "returns": [row["return"] for row in rows],
                          "reference_counts": by_reference, "width_ratio": task["width_ratio"],
                          "units": [{"eligible": row["competence"]["eligible_for_sensitivity_claim"],
                                     "environment_verdict": row["verdict_env"],
                                     "weight_verdict": row["verdict_weight"]} for row in rows],
                          "weight_reference_sd": task["weight_null"]["std"],
                          "environment_reference_sd": task["env_null"]["std"]})
        controls[str(directory.relative_to(reports))] = tasks
    larger = reports / "n99/sanity_test.json"
    primary["larger_bank_market"] = ([comparison(row) for row in read(larger)["real_results_per_seed"]]
                                      if larger.exists() else None)
    return {"market": market, "headline": primary, "conditional_calibration": rates,
            "controls": controls, "state_interventions": state_statistics(
                reports, market_returns=market_returns, market_epochs=market_epochs),
            "steering": steering_statistics(reports), "objective_controls": objective_statistics(reports),
            "undefined_convention": "zero-scale standardized distances are null, including a legacy zero score",
            "seed_spread_convention": "sample standard deviation across training-seed means, ddof=1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-quick", action="store_true")
    args = ap.parse_args()
    reports = args.run / "reports"
    require(not args.out.resolve().is_relative_to(reports.resolve()),
            "publication output must not rewrite immutable experiment artifacts")
    verification = verify(args.run, args.corpus, allow_quick=args.allow_quick)
    payload = {"schema_version": 1, "verification": verification,
               "input_manifest_sha256": digest(reports / "run_manifest.json"),
               "statistics_source_sha256": digest(Path(__file__)), **derive(reports)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}; scientific budgets checked: {verification['scientific_budgets_checked']}")


if __name__ == "__main__":
    main()
