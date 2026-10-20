"""Record immutable inputs and validate outputs of one reproduction run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import EpisodeBatch

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
        for key in ("corpus_sha256", "commit", "quick", "source_sha256"):
            if previous[key] != manifest[key]:
                raise ValueError(f"run input changed since start: {key}")
    if args.phase == "complete":
        validate_evaluation(args.out)
        manifest["artifacts"] = {
            str(p.relative_to(args.out)): digest(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file() and p != path
        }
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"{args.phase}: {path}")


if __name__ == "__main__":
    main()
