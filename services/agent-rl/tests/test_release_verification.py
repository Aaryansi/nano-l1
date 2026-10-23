"""A completed smoke run is not evidence that the release protocol was run."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_market_rollouts import audit
from verify_release import require_release_mode, verify_full_budgets, verify_hashes
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.features import N_FEATURES
from nano_rl.env.synthetic import make_learnable_corpus
from nano_rl.explain.rollout import VectorizedRollout, greedy_policy


@pytest.mark.parametrize("manifest,allow,error", [
    ({"phase": "started", "quick": False, "tracked_source_dirty": False}, False, "not completed"),
    ({"phase": "complete", "quick": True, "tracked_source_dirty": False}, False, "quick runs"),
    ({"phase": "complete", "quick": False, "tracked_source_dirty": True}, False, "modified"),
    ({"phase": "complete", "quick": False, "tracked_source_dirty": True}, True, "modified"),
])
def test_unfinished_quick_or_modified_runs_cannot_be_selected(manifest, allow, error):
    with pytest.raises(ValueError, match=error):
        require_release_mode(manifest, allow)


def test_quick_mode_requires_explicit_execution_only_override():
    require_release_mode({"phase": "complete", "quick": True, "tracked_source_dirty": True}, True)


def test_hash_validation_rejects_changed_or_escaping_artifacts(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("original")
    records = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_hashes(tmp_path, records)
    path.write_text("altered")
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_hashes(tmp_path, records)
    with pytest.raises(ValueError, match="escapes"):
        verify_hashes(tmp_path, {"../outside": "unused"})


def test_relabeling_smoke_as_full_does_not_bypass_training_budget(tmp_path):
    checkpoints = {f"ppo/seed{k}.pt": "unused" for k in range(5)} | {
        f"ppo_nofric/seed{k}_nofric.pt": "unused" for k in range(5)}
    manifest = {"checkpoints": checkpoints, "episodes": 6425, "steps": 14,
                "splits": {name: {"episodes": count} for name, count in
                           (("train", 3854), ("validation", 1283), ("test", 1284))}}
    path = tmp_path / "checkpoints/ppo/seed0_log.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"updates": [0, 1]}))
    with pytest.raises(ValueError, match="update budget differs"):
        verify_full_budgets(tmp_path, manifest)


def test_trajectory_auditor_checks_both_objectives_and_rejects_altered_saved_return(tmp_path):
    batch = make_learnable_corpus(n_episodes=40, n_steps=3, seed=7)
    corpus = tmp_path / "corpus.npz"
    np.savez_compressed(corpus, **{field.name: getattr(batch, field.name)
                                 for field in fields(batch) if not field.name.startswith("_")})
    split = walk_forward_split(batch)
    raw = {"open_epoch": split.test.open_epoch}
    for index, directory in enumerate(("ppo", "ppo_nofric")):
        checkpoint = tmp_path / f"checkpoints/{directory}/seed0.pt"
        checkpoint.parent.mkdir(parents=True)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=index))
        agent.save(str(checkpoint))
        values = VectorizedRollout(split.test, normalizer=split.normalizer,
                                   costs_enabled=directory == "ppo").run(greedy_policy(agent))
        for key, suffix in (("returns", "pnl"), ("trades", "trades"), ("fees", "fees")):
            raw[f"{directory}_{suffix}"] = values[key][None, :]
    (tmp_path / "reports").mkdir()
    path = tmp_path / "reports/evaluation_returns.npz"
    np.savez_compressed(path, **raw)
    result = audit(corpus, tmp_path)
    assert len(result["checkpoints"]) == 2
    assert all(row["actions_checked"] == len(split.test) * 3 for row in result["checkpoints"])
    view = tmp_path / "view"
    view.mkdir()
    (view / "checkpoints").symlink_to(tmp_path / "checkpoints", target_is_directory=True)
    (view / "reports").symlink_to(tmp_path / "reports", target_is_directory=True)
    assert audit(corpus, view) == result
    original = raw["ppo_pnl"]
    raw["ppo_pnl"] = original[:, :, None]
    np.savez_compressed(path, **raw)
    with pytest.raises(ValueError, match="saved pnl shape"):
        audit(corpus, tmp_path)
    raw["ppo_pnl"] = original
    raw["ppo_pnl"][0, 0] += 1.
    np.savez_compressed(path, **raw)
    with pytest.raises(ValueError, match="saved pnl"):
        audit(corpus, tmp_path)
