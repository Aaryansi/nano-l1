"""Verify immutable run evidence before using it in the manuscript.

Default mode requires the declared full release budgets. --allow-quick permits
execution checks and never establishes that scientific budgets were met.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_manifest import (digest, validate_control_returns, validate_evaluation,
                          validate_stability_returns)
from portable_validation import (ABSOLUTE_TOLERANCE, validate_primary_endpoints,
                                 validate_reference_bootstraps, validate_horizon_returns)

ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def verify_hashes(root: Path, records: dict) -> None:
    for name, expected in records.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), f"artifact escapes its root: {name}")
        require(path.is_file() and digest(path) == expected, f"content hash mismatch: {name}")


def require_release_mode(manifest, allow_quick):
    require(manifest["phase"] == "complete", "run has not completed")
    require(allow_quick or not manifest["quick"], "quick runs cannot support manuscript results")
    require((allow_quick and manifest["quick"]) or not manifest["tracked_source_dirty"],
            "full release source was modified")


def verify_sources(manifest):
    if manifest["quick"]:
        verify_hashes(ROOT, manifest["source_sha256"])
        return
    require(not manifest["tracked_source_dirty"], "full release source was modified")
    commit = manifest["commit"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "training source requires a full commit identifier")
    tree = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.splitlines()
    expected_paths = {name for name in tree if Path(name).suffix in {".py", ".sh", ".txt", ".yml"}}
    require(set(manifest["source_sha256"]) == expected_paths,
            "recorded source membership differs from the training commit")
    for name, expected in manifest["source_sha256"].items():
        data = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
        require(hashlib.sha256(data).hexdigest() == expected, f"training source hash mismatch: {name}")
        # Reconstruction uses the same numerical implementation as training.
        if name.startswith("services/agent-rl/nano_rl/") or name in {
                "services/agent-rl/scripts/run_manifest.py", "services/agent-rl/scripts/bootstrap_z.py",
                "services/agent-rl/scripts/calibrate_detection.py", "services/agent-rl/scripts/horizon_scaling.py",
                "services/agent-rl/scripts/generalize_gym.py"}:
            require(digest(ROOT / name) == expected, f"reconstruction source differs: {name}")


def verify_full_budgets(run: Path, manifest):
    reports = run / "reports"
    expected = {f"ppo/seed{k}.pt" for k in range(5)} | {
        f"ppo_nofric/seed{k}_nofric.pt" for k in range(5)}
    require(set(manifest["checkpoints"]) == expected, "require all five market seeds for both objectives")
    require(manifest["episodes"] == 6425 and manifest["steps"] == 14,
            "full release requires the pinned market population")
    require([manifest["splits"][name]["episodes"] for name in ("train", "validation", "test")]
            == [3854, 1283, 1284], "full release split counts differ")
    for directory, suffix in (("ppo", ""), ("ppo_nofric", "_nofric")):
        for seed in range(5):
            log = load(run / f"checkpoints/{directory}/seed{seed}{suffix}_log.json")
            require(log["updates"] == list(range(100)), "market training update budget differs")
    for name, count in (("sanity_test.json", 24), ("n99/sanity_test.json", 99)):
        headline = load(reports / name)
        require(len(headline["null_spans"]) == count and headline["training_updates"] == 40
                and headline["synthetic_episodes"] == 1200
                and headline["endpoint_evaluation_episodes"] == 1284
                and len(headline["real_results_per_seed"]) == 5,
                f"headline budget differs: {name}")
    calibration = load(reports / "detection_calibration.json")
    require(calibration["updates"] == 40 and len(calibration["null_spans"]) == 24
            and [row["signal_strength"] for row in calibration["cases"]] == [0., .2, 1.]
            and all(row["n_trials"] == len(row["trials"]) == 30 for row in calibration["cases"]),
            "conditional calibration budget differs")
    require(calibration["reference_json_sha256"] == digest(reports / "sanity_test.json"),
            "calibration reference source differs from the headline bank")
    controls = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1",
                "FrozenLake-v1", "CliffWalking-v0"}
    for directory in (reports, reports / "seeds", reports / "resample"):
        tasks = load(directory / "generalize_gym.json")
        require({row["env_id"] for row in tasks} == controls and len(tasks) == 6,
                "control task membership differs")
        for task in tasks:
            require(task["trainer"] == "sb3" and task["steps_per_agent"] == 400000
                    and task["null_steps"] == 30000 and task["attribution_episodes"] == 20
                    and task["competence_episodes"] == 25
                    and len(task["env_null"]["spans"]) == len(task["weight_null"]["spans"]) == 24,
                    f"control budget differs: {task['env_id']}")
            if directory.name == "seeds":
                require(task["unit"] == "independent_agent" and task["agent_seeds"] == 5
                        and {row["seed"] for row in task["checkpoints"]} == set(range(5)),
                        "independent control seed membership differs")
    steering = load(reports / "steering.json")
    for name in ("real_market", "learnable_synthetic"):
        require(20. in [row["coef"] for row in steering[name]], "fixed steering penalty missing")
        require(all(len(row["pnl"]) == 10 for row in steering[name]), "steering seed count differs")
    objectives = load(reports / "positive_control.json")
    require([row["updates"] for row in objectives["budget_sweep"]] == [40, 100],
            "objective-control budgets differ")
    require(all(task["n_seeds"] == len(task["per_seed"]) == 5
                for row in objectives["budget_sweep"] for task in row["tasks"]),
            "objective-control seed count differs")
    horizon = load(reports / "horizon_scaling.json")
    require(horizon["training_updates"] == 30 and horizon["evaluation_episodes"] == 600
            and [row["n_steps"] for row in horizon["rows"]] == [4, 7, 14, 28, 56]
            and all(len(row["env_spans"]) == len(row["weight_spans"]) == 16 for row in horizon["rows"]),
            "horizon budget differs")


def verify(run: Path, corpus: Path, *, allow_quick=False) -> dict:
    reports = run / "reports"
    manifest = load(reports / "run_manifest.json")
    require_release_mode(manifest, allow_quick)
    require(digest(corpus) == manifest["corpus_sha256"], "corpus content hash mismatch")
    if not manifest["quick"]:
        pinned = load(ROOT / "data/releases/market_snapshot_v1/manifest.json")
        require(manifest["corpus_sha256"] == pinned["files"]["corpus.npz"],
                "full release requires the pinned corpus")
    verify_sources(manifest)
    for directory, key in ((run / "checkpoints", "checkpoints"),
                           (run / "checkpoints", "training_logs"), (reports, "artifacts")):
        verify_hashes(directory, manifest[key])
    actual = {str(path.relative_to(reports)) for path in reports.rglob("*")
              if path.is_file() and path != reports / "run_manifest.json"}
    require(actual == set(manifest["artifacts"]), "report artifact membership differs")
    actual_checkpoints = {str(path.relative_to(run / "checkpoints"))
                          for path in (run / "checkpoints").rglob("seed*.pt")}
    require(actual_checkpoints == set(manifest["checkpoints"]), "checkpoint membership differs")
    for validator in (validate_evaluation, validate_control_returns, validate_stability_returns):
        validator(reports)
    differences = []
    validate_primary_endpoints(reports, differences)
    validate_horizon_returns(reports, differences)
    validate_reference_bootstraps(reports, differences)
    if differences:
        error, context, actual, recorded = max(differences)
        print(f"accepted CPU roundoff in {len(differences)} numeric fields; maximum {error:.3g} "
              f"at {context}: {actual!r} versus {recorded!r}", file=sys.stderr)
    parity = load(reports / "rollout_parity.json")
    require(parity["corpus_sha256"] == manifest["corpus_sha256"], "parity corpus differs")
    require(parity["audit_script_sha256"] == digest(Path(__file__).with_name("audit_market_rollouts.py")),
            "parity auditor source differs")
    require(len(parity["checkpoints"]) == len(manifest["checkpoints"])
            and {row["checkpoint"] for row in parity["checkpoints"]} == set(manifest["checkpoints"]),
            "parity checkpoint membership differs")
    count, steps = manifest["splits"]["test"]["episodes"], manifest["steps"]
    for row in parity["checkpoints"]:
        require(row["checkpoint_sha256"] == manifest["checkpoints"][row["checkpoint"]]
                and row["episodes"] == count and row["steps"] == steps
                and row["actions_checked"] == row["acted_on_observations_checked"] == count * steps
                and row["maximum_observation_error"] == 0 and row["matches_saved_evaluation"] is True
                and set(row["maximum_accounting_errors"]) == {"returns", "trades", "fees", "final_position"}
                and all(np.isfinite(value) and 0 <= value <= 1e-10
                        for value in row["maximum_accounting_errors"].values()),
                f"invalid trajectory parity evidence: {row['checkpoint']}")
    if not manifest["quick"]:
        verify_full_budgets(run, manifest)
    return {"source_commit": manifest["commit"], "artifacts_checked": len(manifest["artifacts"]),
            "checkpoints_checked": len(manifest["checkpoints"]),
            "scientific_budgets_checked": not manifest["quick"],
            "reconstruction_absolute_tolerance": ABSOLUTE_TOLERANCE,
            "verification_script_sha256": digest(Path(__file__)),
            "reconstruction_script_sha256": digest(Path(__file__).with_name("portable_validation.py")),
            "scope": "content hashes, recorded source, declared budgets, primary raw-summary reconstruction and complete market trajectory parity"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--allow-quick", action="store_true")
    args = ap.parse_args()
    print(json.dumps(verify(args.run, args.corpus, allow_quick=args.allow_quick), indent=2))


if __name__ == "__main__":
    main()
