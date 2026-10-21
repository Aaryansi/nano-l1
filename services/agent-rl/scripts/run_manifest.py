"""Record immutable inputs and validate outputs of one reproduction run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.explain.sanity import test_span_against_null
from nano_rl.explain.rank import rank_comparisons
from nano_rl.metrics import paired_bootstrap_p_value, cluster_bootstrap_ci, cluster_bootstrap_p_value

ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(corpus: Path, checkpoints: Path, phase: str, quick: bool) -> dict:
    batch = EpisodeBatch.load(str(corpus))
    split = walk_forward_split(batch)
    files = sorted(checkpoints.rglob("seed*.pt"))
    if phase == "complete" and not files:
        raise ValueError("completed run has no checkpoints")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "diff", "HEAD", "--quiet"], cwd=ROOT)
    packages = {name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "gymnasium", "scikit-learn", "stable-baselines3")}
    source_paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    sources = {name: digest(ROOT / name) for name in sorted(set(source_paths))
               if (ROOT / name).is_file() and Path(name).suffix in {".py", ".sh", ".txt", ".yml"}}
    return {
        "schema_version": 1, "phase": phase, "quick": quick,
        "commit": git.stdout.strip(), "tracked_source_dirty": dirty.returncode != 0,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "corpus_sha256": digest(corpus),
        "runtime": {"torch_threads": torch.get_num_threads(),
                    "environment": {key: os.environ.get(key) for key in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")}},
        "source_sha256": sources,
        "corpus_file": corpus.name, "episodes": len(batch), "steps": batch.n_steps,
        "splits": {name: {"episodes": len(part), "open_epoch": part.open_epoch.tolist()}
                   for name, part in (("train", split.train), ("validation", split.val),
                                      ("test", split.test))},
        "checkpoints": {str(p.relative_to(checkpoints)): digest(p) for p in files},
        "training_logs": {str(p.relative_to(checkpoints)): digest(p)
                          for p in sorted(checkpoints.rglob("*.json"))},
    }


def validate_evaluation(out: Path) -> None:
    results = out / "results.json"
    if not results.exists():
        return  # explanation-only runs do not recompute the policy table
    artifact = json.loads(results.read_text())
    with np.load(out / "evaluation_returns.npz", allow_pickle=False) as raw:
        pnl = raw["ppo_pnl"]
        if not np.isfinite(pnl).all() or pnl.shape[1] != artifact["test_episodes"]:
            raise ValueError("raw PPO returns have invalid values or episode count")
        summary = artifact["ppo_across_seeds"]
        if not np.isclose(pnl.mean(), summary["mean_pnl_mean"], rtol=0, atol=1e-9):
            raise ValueError("PPO mean does not match raw episode returns")
        if not np.isclose(pnl.mean(axis=1).std(), summary["mean_pnl_std"], rtol=0, atol=1e-9):
            raise ValueError("PPO seed spread does not match raw episode returns")
        for row, episode_pnl in zip(artifact["ppo_per_seed"], pnl, strict=True):
            if not np.isclose(row["mean_pnl"], episode_pnl.mean(), rtol=0, atol=1e-9):
                raise ValueError("per-seed mean does not match raw episode returns")
        if "paired_bootstrap_design" in artifact:
            design = artifact["paired_bootstrap_design"]
            blocks = np.floor(raw["open_epoch"] / 86400).astype(int)
            flat = raw["baseline_always-flat_pnl"]
            for name, recorded in artifact["paired_bootstrap_p_vs_flat"].items():
                values = pnl.mean(axis=0) if name == "ppo" else raw[f"baseline_{name}_pnl"]
                actual = paired_bootstrap_p_value(values, flat, design["n_boot"], design["seed"], block_ids=blocks)
                if not np.isclose(actual, recorded, rtol=0, atol=1e-12):
                    raise ValueError("market comparison differs from raw UTC-day returns")
            primary = artifact["ppo_seed_day_bootstrap"]
            zeros = [np.zeros_like(row) for row in pnl]
            probability = cluster_bootstrap_p_value(list(pnl), zeros, block_ids=blocks)
            interval = cluster_bootstrap_ci(list(pnl), zeros, block_ids=blocks)
            if not np.isclose(primary["p_value"], probability, rtol=0, atol=1e-12, equal_nan=True) or not np.allclose(
                    primary["ci95"], interval, rtol=0, atol=1e-12):
                raise ValueError("primary market uncertainty differs from raw seed/day returns")


def validate_control_returns(out: Path) -> None:
    def endpoint(raw, expected, count):
        full, masked = (np.asarray(raw[key], dtype=float) for key in ("full", "masked"))
        if full.shape != masked.shape or full.shape != (count,) or not (
                np.isfinite(full).all() and np.isfinite(masked).all()):
            raise ValueError("invalid control endpoint returns")
        if len(raw["episode_seeds"]) != count:
            raise ValueError("invalid control episode identities")
        span = float(full.mean() - masked.mean())
        if not np.isclose(span, expected, rtol=0, atol=1e-9) or not np.isclose(
                span, raw["span"], rtol=0, atol=1e-9):
            raise ValueError("control span differs from raw returns")
        return float(full.mean())

    for path in sorted(out.rglob("generalize_gym.json")):
        tasks = json.loads(path.read_text())
        for task in tasks:
            count = task["attribution_episodes"]
            for key in ("weight_null", "env_null"):
                reference = task[key]
                spans = reference["spans"]
                returns = [endpoint(raw, span, count) for raw, span in
                           zip(reference["rollouts"], spans, strict=True)]
                if not np.isclose(np.mean(spans), reference["mean"], rtol=0, atol=1e-9) or not np.isclose(
                        np.std(spans, ddof=1), reference["std"], rtol=0, atol=1e-9):
                    raise ValueError("control reference summary differs from raw returns")
                if key == "weight_null":
                    random = task["random_init_return"]
                    if not np.allclose(returns, random["returns"], rtol=0, atol=1e-9):
                        raise ValueError("random-policy returns differ from endpoint records")
                    if not np.isclose(np.mean(returns), random["mean"], rtol=0, atol=1e-9) or not np.isclose(
                            np.std(returns, ddof=1), random["std"], rtol=0, atol=1e-9):
                        raise ValueError("random-policy summary differs from raw returns")
            for row in task["checkpoints"]:
                endpoint(row["span_rollouts"], row["span"], count)
                values = np.asarray(row["return_rollouts"], dtype=float)
                if values.shape != (task["competence_episodes"],) or not np.isfinite(values).all() or len(values) != len(row["return_episode_seeds"]) or not np.isclose(
                        values.mean(), row["return"], rtol=0, atol=1e-9):
                    raise ValueError("control competence return differs from raw returns")
        derived = path.with_name("null_width_conjecture.json")
        if derived.exists():
            payload = json.loads(derived.read_text())
            if payload["env_ids"] != [task["env_id"] for task in tasks] or not np.allclose(
                    payload["random_return_sd"], [task["random_init_return"]["std"] for task in tasks], rtol=0, atol=1e-9) or not np.allclose(
                    payload["weight_null_sd"], [task["weight_null"]["std"] for task in tasks], rtol=0, atol=1e-9):
                raise ValueError("control width figure inputs differ from raw returns")


def validate_primary_endpoints(out: Path) -> None:
    """Reconstruct primary/reference spans and calibration decisions from episodes."""
    def endpoint(raw, key, count):
        full, masked, identities = (raw[f"{key}_{suffix}"] for suffix in ("full", "masked", "open_epoch"))
        if any(values.shape != (count,) or not np.isfinite(values).all()
               for values in (full, masked, identities)) or len(np.unique(identities)) != count:
            raise ValueError("invalid primary endpoint returns or episode identities")
        return float(full.mean() - masked.mean()), float(full.mean())

    def comparison(raw, key, count, nulls, recorded):
        span, mean = endpoint(raw, key, count)
        if test_span_against_null(span, nulls).as_dict() != recorded:
            raise ValueError("primary comparison differs from raw endpoint returns")
        return mean

    for path in sorted(out.rglob("sanity_test.json")):
        payload = json.loads(path.read_text())
        with np.load(path.parent / payload["endpoint_returns_file"], allow_pickle=False) as raw:
            nulls = [endpoint(raw, key, payload["synthetic_episodes"])[0]
                     for key in payload["reference_endpoint_keys"]]
            if nulls != payload["null_spans"]:
                raise ValueError("primary reference spans differ from raw endpoint returns")
            for name, key in payload["observed_endpoint_keys"].items():
                count = payload["endpoint_evaluation_episodes"] if name == "real_market" else payload["synthetic_episodes"]
                comparison(raw, key, count, nulls, payload["results"][name])
            for key, span, result in zip(payload["real_endpoint_keys"], payload["real_spans_per_seed"],
                                         payload["real_results_per_seed"], strict=True):
                comparison(raw, key, payload["endpoint_evaluation_episodes"], nulls, result)
                if result["statistic"] != span:
                    raise ValueError("primary market span differs from per-seed result")

    from calibrate_detection import summarize
    for path in sorted(out.rglob("detection_calibration.json")):
        payload = json.loads(path.read_text())
        with np.load(path.parent / payload["endpoint_returns_file"], allow_pickle=False) as raw:
            nulls = [endpoint(raw, key, payload["synthetic_episodes"])[0]
                     for key in payload["reference_endpoint_keys"]]
            if nulls != payload["null_spans"]:
                raise ValueError("calibration reference spans differ from raw endpoint returns")
            for case in payload["cases"]:
                for row in case["trials"]:
                    mean = comparison(raw, row["endpoint_key"], row["data"]["evaluation_episodes"], nulls, row["result"])
                    if mean != row["return"]:
                        raise ValueError("calibration return differs from raw endpoint returns")
                    if "rank_comparators" in row and row["rank_comparators"] != rank_comparisons(row["result"]["statistic"], nulls):
                        raise ValueError("calibration rank comparison differs from raw endpoint returns")
                if any(case[key] != value for key, value in summarize(case["trials"]).items()):
                    raise ValueError("calibration rate differs from raw endpoint decisions")


def validate_horizon_returns(out: Path) -> None:
    for path in sorted(out.rglob("horizon_scaling.json")):
        payload = json.loads(path.read_text())
        count = payload["evaluation_episodes"]
        with np.load(path.parent / payload["endpoint_returns_file"], allow_pickle=False) as raw:
            for row in payload["rows"]:
                widths = {}
                for arm in ("weight", "env"):
                    spans = []
                    for key in row["endpoint_keys"][arm]:
                        full, masked, ids = [raw[f"{key}_{suffix}"] for suffix in ("full", "masked", "open_epoch")]
                        if any(a.shape != (count,) or not np.isfinite(a).all() for a in (full, masked, ids)) or len(np.unique(ids)) != count:
                            raise ValueError("invalid horizon endpoint returns or identities")
                        spans.append(float(full.mean() - masked.mean()))
                    widths[arm] = float(np.std(spans, ddof=1))
                    if spans != row[f"{arm}_spans"] or widths[arm] != row[f"{arm}_std"]:
                        raise ValueError("horizon reference differs from raw endpoint returns")
                ratio = widths["weight"] / widths["env"] if widths["env"] > 1e-12 else None
                if ratio != row["ratio"]:
                    raise ValueError("horizon width ratio differs from raw endpoint returns")
        from horizon_scaling import fit_power_law
        x = np.array([row["n_steps"] for row in payload["rows"]], dtype=float)
        for arm in ("weight", "env"):
            y = np.array([row[f"{arm}_std"] for row in payload["rows"]])
            slope, quality = fit_power_law(x, y)
            if (payload.get(f"alpha_{arm}", slope) != slope
                    or payload.get(f"r2_{arm}", quality) != quality):
                raise ValueError("horizon fit differs from reconstructed reference widths")
            if "fit_horizons" in payload and payload["fit_horizons"][arm] != x[y > 0].tolist():
                raise ValueError("horizon fit membership differs from reference widths")


def validate_stability_returns(out: Path) -> None:
    import itertools
    for path in sorted(out.rglob("stability.json")):
        payload = json.loads(path.read_text())
        with np.load(path.parent / payload["performance_returns_file"], allow_pickle=False) as raw:
            pnl, epochs = raw["returns"], raw["open_epoch"]
            if (pnl.ndim != 2 or pnl.shape[0] != payload["n_seeds"] or pnl.shape[1:] != epochs.shape
                    or not np.isfinite(pnl).all() or not np.isfinite(epochs).all()
                    or len(np.unique(epochs)) != len(epochs)):
                raise ValueError("invalid raw stability performance returns")
            blocks = np.floor(epochs / 86400).astype(int)
            pairs = list(itertools.combinations(range(len(pnl)), 2))
            ps = [paired_bootstrap_p_value(pnl[i], pnl[j], payload["performance_bootstrap_resamples"],
                                          payload["performance_bootstrap_seed"], block_ids=blocks) for i, j in pairs]
            if (not np.allclose(pnl.mean(axis=1), payload["seed_pnl"], rtol=0, atol=1e-12)
                    or not np.allclose(ps, payload["performance_pvalues"], rtol=0, atol=1e-12)
                    or payload["performance_pairs"] != [list(pair) for pair in pairs]
                    or payload["performance_utc_day_blocks"] != len(np.unique(blocks))
                    or payload["n_pairs_without_detected_difference"] != sum(p >= .05 for p in ps)):
                raise ValueError("stability comparison differs from raw UTC-day returns")
            evaluation = path.with_name("evaluation_returns.npz")
            if evaluation.exists():
                with np.load(evaluation, allow_pickle=False) as evaluated:
                    if not np.array_equal(epochs, evaluated["open_epoch"]) or not np.allclose(
                            pnl, evaluated["ppo_pnl"], rtol=0, atol=1e-9):
                        raise ValueError("stability and primary evaluations disagree")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", choices=("started", "complete"), required=True)
    ap.add_argument("--quick", type=int, choices=(0, 1), default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.corpus, args.checkpoints, args.phase, bool(args.quick))
    path = args.out / "run_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("corpus_sha256", "commit", "quick", "source_sha256", "packages", "runtime"):
            if previous[key] != manifest[key]:
                raise ValueError(f"run input changed since start: {key}")
    if args.phase == "complete":
        validate_evaluation(args.out)
        validate_control_returns(args.out)
        validate_primary_endpoints(args.out)
        validate_horizon_returns(args.out)
        validate_stability_returns(args.out)
        manifest["artifacts"] = {
            str(p.relative_to(args.out)): digest(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file() and p != path
        }
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"{args.phase}: {path}")


if __name__ == "__main__":
    main()
