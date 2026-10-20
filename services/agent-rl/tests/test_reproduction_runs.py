"""A reproduction must identify and validate the outputs it actually produced."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("run_manifest", ROOT / "services/agent-rl/scripts/run_manifest.py")
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def config(*args, **overrides):
    env = {k: v for k, v in os.environ.items()
           if k not in {"RUN_ROOT", "REPORTS", "FIGURES", "CHECKPOINTS", "CORPUS"}}
    env.update({k: str(v) for k, v in overrides.items()})
    return subprocess.run(["bash", str(ROOT / "reproduce.sh"), *args, "--print-config"],
                          env=env, capture_output=True, text=True)


def test_quick_runs_have_unique_isolated_paths():
    a, b = [json.loads(config("--quick").stdout) for _ in range(2)]
    assert a["run_root"] != b["run_root"]
    for key in ("reports", "figures", "runs", "runs_nofric"):
        assert Path(a[key]).is_relative_to(a["run_root"])
        assert a[key] != str(ROOT / "reports")


def test_explicit_paths_are_used_together(tmp_path):
    result = config("--quick", RUN_ROOT=tmp_path / "release")
    assert result.returncode == 0
    paths = json.loads(result.stdout)
    assert paths["runs"] == str(tmp_path / "release/checkpoints/ppo")
    assert paths["reports"] == str(tmp_path / "release/reports")


def test_smoke_run_cannot_claim_publication_verification():
    result = config("--quick", "--verify-paper")
    assert result.returncode != 0
    assert "cannot verify publication" in result.stdout


def test_explain_only_requires_explicit_existing_checkpoints(tmp_path):
    result = config("--explain-only", CHECKPOINTS=tmp_path / "missing")
    assert result.returncode != 0
    assert "requires CHECKPOINTS" in result.stdout


def test_verifier_does_not_fall_back_to_published_artifacts(tmp_path):
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "services/agent-rl/scripts/verify_paper_numbers.py"),
                             "--reports", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "results.json: artifact missing" in result.stdout


def test_raw_return_validation_catches_stale_summary(tmp_path):
    pnl = np.array([[1., 2., 3.], [2., 3., 4.]])
    np.savez(tmp_path / "evaluation_returns.npz", ppo_pnl=pnl)
    payload = {"test_episodes": 3, "ppo_across_seeds": {
        "mean_pnl_mean": 2.5, "mean_pnl_std": .5},
        "ppo_per_seed": [{"mean_pnl": 2.}, {"mean_pnl": 3.}]}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))
    manifest.validate_evaluation(tmp_path)
    payload["ppo_across_seeds"]["mean_pnl_mean"] = -10
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mean does not match"):
        manifest.validate_evaluation(tmp_path)


def test_hash_identifies_content_not_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same content")
    b.write_bytes(a.read_bytes())
    assert manifest.digest(a) == manifest.digest(b)
    b.write_bytes(b"changed")
    assert manifest.digest(a) != manifest.digest(b)


def test_default_reproduction_selects_pinned_corpus():
    paths = json.loads(config().stdout)
    assert paths["corpus"] == str(ROOT / "data/releases/market_snapshot_v1/corpus.npz")


def test_pinned_snapshot_and_tamper_detection(tmp_path):
    import shutil
    spec = importlib.util.spec_from_file_location("verify_snapshot", ROOT / "services/agent-rl/scripts/verify_snapshot.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    source = ROOT / "data/releases/market_snapshot_v1"
    assert verifier.verify_snapshot(source)["episodes"] == 6425
    target = tmp_path / "snapshot"
    shutil.copytree(source, target)
    with (target / "corpus.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier.verify_snapshot(target)


def test_control_artifact_validation_catches_modified_episode_returns(tmp_path):
    def endpoint(span):
        return {"span": span, "full": [span, span], "masked": [0., 0.],
                "episode_seeds": [0, 1]}
    reference = {"spans": [1., 3.], "mean": 2., "std": np.sqrt(2.),
                 "rollouts": [endpoint(1.), endpoint(3.)]}
    payload = [{"env_id": "test", "attribution_episodes": 2, "competence_episodes": 2,
                "weight_null": reference, "env_null": reference,
                "random_init_return": {"returns": [1., 3.], "mean": 2., "std": np.sqrt(2.)},
                "checkpoints": [{"span": 2., "span_rollouts": endpoint(2.), "return": 2.,
                                 "return_rollouts": [1., 3.], "return_episode_seeds": [0, 1]}]}]
    path = tmp_path / "generalize_gym.json"
    path.write_text(json.dumps(payload))
    manifest.validate_control_returns(tmp_path)
    payload[0]["checkpoints"][0]["span_rollouts"]["masked"][0] = 1.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="control span differs"):
        manifest.validate_control_returns(tmp_path)
