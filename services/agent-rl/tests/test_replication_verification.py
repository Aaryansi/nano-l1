"""Supplemental study gates require complete evidence and replay-linked identity."""
import copy
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_reference_banks as banks
import objective_study as objective
from nano_rl.env.synthetic import make_learnable_corpus
from study_archive import pack
from verify_replications import bank_statistics, check_output, derive

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("replication")
    corpus, original = root / "corpus.npz", root / "original"
    batch = make_learnable_corpus(n_episodes=48, n_steps=6, signal_strength=.8, seed=19)
    np.savez_compressed(corpus, **{f.name: getattr(batch, f.name) for f in fields(batch)
                                  if not f.name.startswith("_") and getattr(batch, f.name) is not None})
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ALLOW_UNRESOLVED_NULLS": "1"}
    subprocess.run([sys.executable, str(ROOT / "services/agent-rl/scripts/positive_control.py"),
                    "--corpus", str(corpus), "--out", str(original), "--n-null", "2",
                    "--seeds", "2", "--budgets", "1", "2"], env=env, check=True, capture_output=True)
    study = root / "objective"
    result = objective.execute(study, corpus, original, quick=True)
    assert result["scientific_budgets_checked"] is False
    bank = root / "bank"
    config = banks.STANDARD | {"banks": 2, "n_null": 2, "trials": 1, "updates": 1,
                               "episodes": 8, "strengths": [0., 1.], "seed": 17, "jobs": 1}
    banks.run(config, bank)
    bundle, proof = root / "bundle", root / "replay.json"
    pack(bank, bundle)
    bank_statistics(bundle, proof, replay=True, allow_quick=True)
    return study, corpus, original, bundle, proof


def test_complete_execution_studies_are_explicitly_insufficient_for_science(completed):
    study, corpus, original, bundle, proof = completed
    result = derive(bundle, proof, study, corpus, original, allow_quick=True)
    assert result["scientific_budgets_checked"] is False
    assert result["reference_banks"]["recorded_checkpoint_replay"]["endpoints_replayed"] is True
    assert result["reference_banks"]["verification"]["endpoints_replayed"] is False
    assert result["objective_replication"]["verification"]["comparison"]["maximum_episode_return_difference"] == 0.
    with pytest.raises(ValueError, match="full declared budgets"):
        objective.verify(study, corpus, original)
    with pytest.raises(ValueError, match="full scientific protocol"):
        derive(bundle, proof, study, corpus, original)
    with pytest.raises(ValueError, match="existing"):
        objective.execute(study, corpus, original, quick=True)


@pytest.mark.parametrize("change", ["phase", "config", "config_type", "logs", "source", "extra", "hash",
                                    "verification", "comparison", "original_contrast"])
def test_altered_objective_provenance_or_comparison_is_rejected(completed, tmp_path, change):
    source, corpus, original, _, _ = completed
    study = tmp_path / "study"
    shutil.copytree(source, study)
    path = study / "objective_study_manifest.json"
    manifest = objective.load(path)
    if change == "phase":
        manifest["phase"] = "started"
    elif change == "config":
        manifest["config"] = copy.deepcopy(objective.STANDARD)
    elif change == "config_type":
        manifest["config"]["seed"] = False
    elif change == "logs":
        manifest["training_logs_archived"] = True
    elif change == "source":
        key = next(iter(manifest["provenance"]["source_sha256"]))
        manifest["provenance"]["source_sha256"][key] = "0" * 64
    elif change == "extra":
        (study / "stale.json").write_text('{}')
    elif change == "hash":
        manifest["artifacts_sha256"]["comparison.json"] = "0" * 64
    elif change == "verification":
        manifest["verification"]["observed_policies"] += 1
    else:
        report = objective.load(study / "comparison.json")
        if change == "comparison":
            report["maximum_episode_return_difference"] = 1.
        else:
            report["contrasts_for_review"][0]["original_recorded_contrast"]["n_defined"] += 1
        objective.write(study / "comparison.json", report)
        manifest["artifacts_sha256"]["comparison.json"] = objective.digest(study / "comparison.json")
    objective.write(path, manifest)
    with pytest.raises(ValueError):
        objective.verify(study, corpus, original, allow_quick=True)


@pytest.mark.parametrize("change", ["manifest", "source", "replay", "count"])
def test_bank_proof_must_identify_this_study_and_replay(completed, tmp_path, change):
    _, _, _, bundle, proof = completed
    altered = json.loads(proof.read_text())
    if change == "manifest":
        altered["study_manifest_sha256"] = "0" * 64
    elif change == "source":
        altered["verification_source_sha256"] = "0" * 64
    elif change == "replay":
        altered["verification"]["endpoints_replayed"] = False
    else:
        altered["verification"]["models_archived"] += 1
    path = tmp_path / "proof.json"
    objective.write(path, altered)
    with pytest.raises(ValueError, match="replay proof differs"):
        bank_statistics(bundle, path, allow_quick=True)


def test_full_objective_run_refuses_dirty_source_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(objective, "source_state", lambda: {"tracked_source_dirty": True, "untracked_source": []})
    with pytest.raises(ValueError, match="clean committed source"):
        objective.execute(tmp_path / "run", tmp_path / "corpus", tmp_path / "original")
    assert not (tmp_path / "run").exists()


def test_replay_proof_cannot_be_written_inside_archive_bundle(completed):
    _, _, _, bundle, _ = completed
    with pytest.raises(ValueError, match="outside immutable evidence"):
        bank_statistics(bundle, bundle / "proof.json", replay=True, allow_quick=True)
    assert not (bundle / "proof.json").exists()


@pytest.mark.parametrize("destination", ["banks/new.json", "objectives/new.json", "original/new.json",
                                         "corpus.npz", "proof.json", "existing.json"])
def test_derived_output_cannot_replace_or_enter_its_evidence(tmp_path, destination):
    (tmp_path / "existing.json").write_text("retained")
    with pytest.raises(ValueError, match="fresh path outside"):
        check_output(tmp_path / destination, *(tmp_path / name for name in
                     ("banks", "objectives", "original", "corpus.npz", "proof.json")))
    assert (tmp_path / "existing.json").read_text() == "retained"
