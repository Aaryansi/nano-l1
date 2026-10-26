"""Run and verify a separately identified replication of objective endpoints.

The public trainer supplies the endpoint evidence. This layer records its
source/runtime, checks every original return/span and seals the complete run.
It does not claim archived training logs or checkpoint replay.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from calibrate_reference_banks import ROOT, source_state, verify_sources
from portable_validation import reconstructed
from run_manifest import digest
from verify_objective_control import verify as verify_endpoints

STANDARD = {"budgets": [40, 100], "n_null": 24, "agent_seeds": 5, "seed": 0}
QUICK = {"budgets": [1, 2], "n_null": 2, "agent_seeds": 2, "seed": 0}
COMPARISON_SCOPE = ("all original episode returns, observed/reference spans, finite comparisons and decisions; "
                    "undefined zero-scale diagnostics normalized to null; original contrasts retained for review")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def compare_with_original(reports, original):
    with np.load(original / "positive_control_returns.npz", allow_pickle=False) as old, np.load(
            reports / "positive_control_returns.npz", allow_pickle=False) as new:
        for key in ("budgets", "open_epoch"):
            require(np.array_equal(old[key], new[key]), "original objective array membership differs")
        require(old["returns"].shape == new["returns"].shape
                and np.isfinite(old["returns"]).all() and np.isfinite(new["returns"]).all(),
                "original objective return shape or finiteness differs")
        maximum = float(np.max(np.abs(old["returns"] - new["returns"])))
        require(np.allclose(old["returns"], new["returns"], rtol=0, atol=1e-9),
                "original objective episode returns did not reproduce")
    old, new = load(original / "positive_control.json"), load(reports / "positive_control.json")
    differences, contrasts = [], []
    for a, b in zip(old["budget_sweep"], new["budget_sweep"], strict=True):
        reconstructed(b["updates"], a["updates"], "objective repeated budget", differences)
        for ta, tb in zip(a["tasks"], b["tasks"], strict=True):
            for key in ("task", "updates", "n_seeds", "return", "span", "fires", "verdict",
                        "return_mean_across_seeds", "span_mean_across_seeds", "null_spans", "null_mean", "null_std"):
                reconstructed(tb[key], ta[key], "objective original summary/" + key, differences)
            for ra, rb in zip(ta["per_seed"], tb["per_seed"], strict=True):
                for key in ("seed", "return", "span", "fires", "verdict"):
                    reconstructed(rb[key], ra[key], "objective original agent/" + key, differences)
                result = copy.deepcopy(ra["result"])
                if result["null_std"] <= 1e-12:
                    result["z_score"] = result["p_normal"] = None
                reconstructed(rb["result"], result, "objective original comparison", differences)
        for key in ("separated", "undecided_tasks"):
            reconstructed(b[key], a[key], "objective original decision/" + key, differences)
        contrasts.append({"updates": a["updates"], "original_recorded_contrast": a["contrast"],
                          "reconstructed_contrast": b["contrast"]})
    return {"original_objective_json_sha256": digest(original / "positive_control.json"),
            "original_objective_returns_sha256": digest(original / "positive_control_returns.npz"),
            "maximum_episode_return_difference": maximum, "absolute_tolerance": 1e-9,
            "accepted_scalar_roundoff_fields": len(differences),
            "maximum_scalar_roundoff": max((row[0] for row in differences), default=0.),
            "contrasts_for_review": contrasts, "scope": COMPARISON_SCOPE}


def verify(run, corpus, original, *, allow_quick=False):
    manifest = load(run / "objective_study_manifest.json")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["phase"] == "complete", "objective study has not completed with a supported schema")
    config = manifest["config"]
    require(set(config) == set(STANDARD) and isinstance(config["budgets"], list)
            and all(type(n) is int for n in config["budgets"])
            and all(type(config[key]) is int for key in ("n_null", "agent_seeds", "seed")),
            "objective study configuration types differ")
    require(config == STANDARD or (allow_quick and config == QUICK),
            "objective study requires the full declared budgets")
    require(manifest["training_logs_archived"] is False and manifest["checkpoints_archived"] is False,
            "objective study cannot claim archived training or checkpoints")
    require(manifest["corpus_sha256"] == digest(corpus), "objective study corpus hash differs")
    provenance = manifest["provenance"]
    verify_sources(provenance, allow_quick)
    for name in ("positive_control.py", "verify_objective_control.py"):
        path = "services/agent-rl/scripts/" + name
        require(digest(ROOT / path) == provenance["source_sha256"][path],
                "objective reconstruction source differs: " + name)
    names = {"comparison.json", "reports/positive_control.json", "reports/positive_control_returns.npz"}
    require(set(manifest["artifacts_sha256"]) == names and
            {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()
             and p != run / "objective_study_manifest.json"} == names,
            "objective study artifact membership differs")
    for name, fingerprint in manifest["artifacts_sha256"].items():
        path = run / name
        require(path.resolve().is_relative_to(run.resolve()) and digest(path) == fingerprint,
                "objective study artifact hash differs")
    result = verify_endpoints(run / "reports", corpus)
    reconstructed(result, manifest["verification"], "objective verification record")
    payload = load(run / "reports/positive_control.json")
    require(payload["seed"] == config["seed"] and payload["n_null"] == config["n_null"]
            and [row["updates"] for row in payload["budget_sweep"]] == config["budgets"]
            and all(task["n_seeds"] == config["agent_seeds"] for row in payload["budget_sweep"]
                    for task in row["tasks"]), "objective study budgets disagree with endpoints")
    comparison = compare_with_original(run / "reports", original)
    reconstructed(comparison, load(run / "comparison.json"), "objective historical comparison")
    return {**result, "scientific_budgets_checked": config == STANDARD,
            "source_commit": provenance["commit"], "comparison": comparison}


def execute(out, corpus, original, *, quick=False):
    torch.set_num_threads(1)
    require(not out.exists(), "refusing existing objective study output")
    require(not any(out.resolve().is_relative_to(ROOT / name) for name in ("docs", "data/releases")),
            "objective study output must be separate from released evidence")
    config = QUICK if quick else STANDARD
    provenance = source_state()
    require(quick or (not provenance["tracked_source_dirty"] and not provenance["untracked_source"]),
            "full objective study requires clean committed source")
    out.mkdir(parents=True)
    manifest = {"schema_version": 1, "phase": "started", "provenance": provenance, "config": config,
                "corpus_sha256": digest(corpus), "training_logs_archived": False, "checkpoints_archived": False,
                "scope": "repeat recorded objective returns/spans with both raw endpoints; no original checkpoint replay"}
    manifest_path = out / "objective_study_manifest.json"
    write(manifest_path, manifest)
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, str(ROOT / "services/agent-rl/scripts/positive_control.py"),
                        "--corpus", str(corpus), "--out", str(out / "reports"), "--n-null", str(config["n_null"]),
                        "--seeds", str(config["agent_seeds"]), "--budgets", *map(str, config["budgets"]),
                        "--seed", str(config["seed"])], check=True,
                       env=os.environ | ({"ALLOW_UNRESOLVED_NULLS": "1"} if quick else {}))
        result = verify_endpoints(out / "reports", corpus)
        write(out / "comparison.json", compare_with_original(out / "reports", original))
        require(source_state() == provenance and digest(corpus) == manifest["corpus_sha256"],
                "objective study source, corpus or runtime changed")
        manifest.update(phase="complete", verification=result, seconds=time.monotonic() - started,
                        artifacts_sha256={p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*"))
                                          if p.is_file() and p != manifest_path})
        write(manifest_path, manifest)
        return verify(out, corpus, original, allow_quick=quick)
    except BaseException as error:
        manifest.update(phase="failed", error=str(error), seconds=time.monotonic() - started)
        write(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/releases/market_snapshot_v1/corpus.npz")
    parser.add_argument("--original", type=Path, default=ROOT / "data/releases/research_v1/reports")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-quick", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = (verify(args.out, args.corpus, args.original, allow_quick=args.allow_quick) if args.verify_only
              else execute(args.out.resolve(), args.corpus.resolve(), args.original.resolve(), quick=args.quick))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
