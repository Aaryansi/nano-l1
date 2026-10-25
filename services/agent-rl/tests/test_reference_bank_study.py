"""Replicated banks retain fresh data, raw endpoints and replayable policies."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_reference_banks as study

ROOT = Path(__file__).resolve().parents[3]


def small_config():
    return study.STANDARD | {"banks": 2, "n_null": 2, "trials": 1, "updates": 1,
                             "episodes": 8, "strengths": [0., 1.], "seed": 17, "jobs": 1}


def test_seed_plan_changes_every_bank_corpus_and_policy_stream():
    config = study.STANDARD | {"jobs": 2}
    plan = study.seed_plan(config)
    assert plan == study.seed_plan(config)
    seeds = [seed for bank in plan for agent in bank["agents"] for seed in agent.values()]
    assert len(seeds) == len(set(seeds)) == 2280 * 4
    assert len(plan) == 20
    tiny = small_config()
    hashes = [study.corpus_digest(study.population(tiny, 0., seed))
              for bank in study.seed_plan(tiny) for row in bank["agents"]
              for seed in (row["train_seed"], row["evaluation_seed"])]
    assert len(hashes) == len(set(hashes))


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank-study")
    output = root / "run"
    result = study.run(small_config(), output)
    assert result["banks_checked"] == 2
    assert result["models_archived"] == 8
    assert result["scientific_budgets_checked"] is False
    return output


def test_execution_study_cannot_verify_science_and_all_saved_models_replay(completed):
    with pytest.raises(ValueError, match="cannot verify the full scientific protocol"):
        study.verify(completed)
    result = study.verify(completed, allow_quick=True, replay=True)
    assert result["endpoints_replayed"] is True
    assert result["models_archived"] == 8
    with pytest.raises(ValueError, match="nonempty"):
        study.run(small_config(), completed)


def alter_hash(output, name):
    path = output / "study_manifest.json"
    manifest = study.read(path)
    manifest["artifacts_sha256"][name] = study.digest(output / name)
    study.write(path, manifest)


@pytest.mark.parametrize("change,error", [
    ("raw", "endpoint span"), ("seed", "seed plan"), ("decision", "bank decision"),
    ("budget", "training budget"), ("archive_membership", "archive membership"),
    ("extra_file", "artifact membership"), ("incomplete", "has not completed"),
    ("false_full", "budget flag"),
    ("source", "execution source changed|historical source hash differs"),
    ("reused_corpus", "fingerprints are invalid or reused"),
])
def test_verifier_rejects_altered_evidence_even_after_file_hash_update(completed, tmp_path, change, error):
    output = tmp_path / "run"
    shutil.copytree(completed, output)
    bank_path = output / "bank_000/bank.json"
    payload = study.read(bank_path)
    if change in {"raw", "archive_membership"}:
        path = output / "bank_000/endpoints.npz"
        with np.load(path) as source:
            raw = {key: source[key] for key in source.files}
        if change == "raw":
            raw["reference_0_full"][0] += 1.
        else:
            raw["unexpected"] = np.zeros(8)
        np.savez_compressed(path, **raw)
        alter_hash(output, "bank_000/endpoints.npz")
    elif change in {"seed", "decision", "reused_corpus"}:
        row = payload["cases"][0]["trials"][0]
        if change == "seed":
            row["policy_seed"] += 1
        elif change == "decision":
            row["methods"]["legacy"]["flagged"] = not row["methods"]["legacy"]["flagged"]
        else:
            row["train_corpus_sha256"] = payload["references"][0]["train_corpus_sha256"]
        study.write(bank_path, payload)
        alter_hash(output, "bank_000/bank.json")
    elif change == "budget":
        name = "bank_000/" + payload["references"][0]["training_log"]
        log = study.read(output / name)
        log["updates"] = []
        study.write(output / name, log)
        alter_hash(output, name)
    else:
        path = output / "study_manifest.json"
        manifest = study.read(path)
        if change == "extra_file":
            extra = output / "bank_000/study_manifest.json"
            study.write(extra, {"unexpected": True})
            manifest["artifacts_sha256"]["bank_000/study_manifest.json"] = study.digest(extra)
        elif change == "incomplete":
            manifest["phase"] = "started"
        elif change == "false_full":
            manifest["scientific_budgets_checked"] = True
        else:
            key = next(iter(manifest["provenance"]["source_sha256"]))
            manifest["provenance"]["source_sha256"][key] = "0" * 64
        study.write(path, manifest)
    with pytest.raises(ValueError, match=error):
        study.verify(output, allow_quick=True)


@pytest.mark.parametrize("dirty", [False, True])
def test_source_hash_tampering_is_rejected_for_clean_and_dirty_runs(tmp_path, monkeypatch, dirty):
    source = tmp_path / "example.py"
    source.write_text("value = 1\n")
    for args in (["init", "--quiet"], ["add", "example.py"],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                  "commit", "--quiet", "-m", "Initial source"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    provenance = {"commit": commit, "tracked_source_dirty": dirty, "untracked_source": [],
                  "source_sha256": {"example.py": study.digest(source)}}
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "source_state", lambda: copy.deepcopy(provenance))
    study.verify_sources(provenance, allow_quick=True)
    altered = copy.deepcopy(provenance)
    altered["source_sha256"]["example.py"] = "0" * 64
    error = "execution source changed" if dirty else "historical source hash differs"
    with pytest.raises(ValueError, match=error):
        study.verify_sources(altered, allow_quick=True)


def test_checkpoint_replay_detects_wrong_policy_after_hash_update(completed, tmp_path):
    output = tmp_path / "run"
    shutil.copytree(completed, output)
    payload = study.read(output / "bank_000/bank.json")
    record = payload["references"][0]
    path = output / "bank_000" / record["checkpoint"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    with np.load(output / "bank_000/endpoints.npz") as raw:
        action = 1 if not np.allclose(raw["reference_0_full"], 0.) else 2
    for tensor in saved["model"].values():
        tensor.zero_()
    saved["model"]["policy_head.bias"][action] = 10.
    torch.save(saved, path)
    alter_hash(output, "bank_000/" + record["checkpoint"])
    study.verify(output, allow_quick=True)
    with pytest.raises(ValueError, match="checkpoint replay differs"):
        study.verify(output, allow_quick=True, replay=True)


def test_released_outputs_are_protected_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    for name in ("data/releases/new", "docs/paper/new"):
        with pytest.raises(ValueError, match="separate from released evidence"):
            study.run(small_config(), tmp_path / name)
        assert not (tmp_path / name).exists()


def test_replay_rejects_nonfinite_checkpoint_weights(completed, tmp_path):
    output = tmp_path / "run"
    shutil.copytree(completed, output)
    payload = study.read(output / "bank_000/bank.json")
    record = payload["references"][0]
    path = output / "bank_000" / record["checkpoint"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved["model"]["value_head.bias"][0] = float("nan")
    torch.save(saved, path)
    alter_hash(output, "bank_000/" + record["checkpoint"])
    with pytest.raises(ValueError, match="non-finite checkpoint"):
        study.verify(output, allow_quick=True, replay=True)


def test_full_study_requires_clean_committed_source_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "source_state", lambda: {
        "tracked_source_dirty": True, "untracked_source": []})
    output = tmp_path / "full"
    with pytest.raises(ValueError, match="committed clean source"):
        study.run(study.STANDARD | {"jobs": 1}, output)
    assert not output.exists()


def test_serial_and_spawned_banks_have_identical_endpoint_arrays(tmp_path):
    script = ROOT / "services/agent-rl/scripts/calibrate_reference_banks.py"
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
                       "ALLOW_UNRESOLVED_NULLS": "1"}
    outputs = []
    for jobs in (1, 2, 4):
        output = tmp_path / str(jobs)
        result = subprocess.run([sys.executable, str(script), "--out", str(output), "--banks", "4",
                                 "--n-null", "2", "--trials", "1", "--strengths", "0",
                                 "--updates", "1", "--episodes", "8", "--seed", "17", "--jobs", str(jobs)], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(output)
    for output in outputs[1:]:
        for index in range(4):
            with np.load(outputs[0] / f"bank_{index:03d}/endpoints.npz") as a, np.load(
                    output / f"bank_{index:03d}/endpoints.npz") as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            assert study.read(outputs[0] / f"bank_{index:03d}/bank.json") == study.read(
                output / f"bank_{index:03d}/bank.json")
            bank = study.read(output / f"bank_{index:03d}/bank.json")
            for row in bank["references"] + [row for case in bank["cases"] for row in case["trials"]]:
                original = torch.load(outputs[0] / f"bank_{index:03d}" / row["checkpoint"],
                                      map_location="cpu", weights_only=False)
                parallel = torch.load(output / f"bank_{index:03d}" / row["checkpoint"],
                                      map_location="cpu", weights_only=False)
                assert original["cfg"] == parallel["cfg"]
                for key in original["model"]:
                    torch.testing.assert_close(original["model"][key], parallel["model"][key], rtol=0, atol=0)
