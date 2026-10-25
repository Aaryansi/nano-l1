"""Future objective comparisons require reconstructible observed/reference endpoints."""
from dataclasses import fields
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nano_rl.env.synthetic import make_learnable_corpus
from run_manifest import digest
from verify_objective_control import verify
import positive_control as producer

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "services/agent-rl/scripts/positive_control.py"
ENV = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
                    "ALLOW_UNRESOLVED_NULLS": "1"}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("objective-evidence")
    batch = make_learnable_corpus(n_episodes=48, n_steps=6, signal_strength=.8, seed=17)
    corpus, output = root / "corpus.npz", root / "reports"
    np.savez_compressed(corpus, **{field.name: getattr(batch, field.name) for field in fields(batch)
                                  if not field.name.startswith("_") and getattr(batch, field.name) is not None})
    result = subprocess.run([sys.executable, str(SCRIPT), "--corpus", str(corpus), "--out", str(output),
                             "--n-null", "3", "--seeds", "2", "--budgets", "1", "2", "--seed", "17"],
                            env=ENV, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output, corpus


def test_every_observed_and_reference_endpoint_reconstructs_without_mutation(completed):
    output, corpus = completed
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    result = verify(output, corpus)
    assert result["budgets_checked"] == 2
    assert result["observed_policies"] == 8
    assert result["reference_policies"] == 12
    with np.load(output / "positive_control_returns.npz") as raw:
        for key in ("returns", "masked_returns", "reference_returns", "reference_masked_returns"):
            count = 3 if key.startswith("reference_") else 2
            assert raw[key].shape == (2, 2, count, result["episodes_per_endpoint"])
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    command = subprocess.run([sys.executable, str(SCRIPT.with_name("verify_objective_control.py")),
                              "--reports", str(output), "--corpus", str(corpus)],
                             env=ENV, capture_output=True, text=True)
    assert command.returncode == 0, command.stdout + command.stderr
    assert json.loads(command.stdout) == result


@pytest.mark.parametrize("change", [
    "returns", "masked_returns", "reference_returns", "reference_masked_returns", "open_epoch",
    "budgets", "task_names", "agent_seeds", "observed_masking_seeds", "reference_policy_seeds",
    "reference_masking_seeds", "reference_blinding_seeds", "extra", "shape", "nonfinite",
])
def test_altered_raw_evidence_is_rejected_even_after_hash_update(completed, tmp_path, change):
    output, corpus = completed
    reports = tmp_path / "reports"
    shutil.copytree(output, reports)
    archive = reports / "positive_control_returns.npz"
    with np.load(archive) as saved:
        raw = {key: saved[key] for key in saved.files}
    if change == "extra":
        raw["unrecorded"] = np.ones(1)
    elif change == "shape":
        raw["masked_returns"] = raw["masked_returns"][..., :1]
    elif change == "nonfinite":
        raw["reference_returns"].flat[0] = np.nan
    elif change == "task_names":
        raw[change] = raw[change][::-1]
    else:
        raw[change].flat[0] += 1
    np.savez_compressed(archive, **raw)
    path = reports / "positive_control.json"
    payload = json.loads(path.read_text())
    payload["endpoint_returns_sha256"] = digest(archive)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="objective|reconstruction differs"):
        verify(reports, corpus)


@pytest.mark.parametrize("change", ["span", "decision", "contrast", "primary_alias", "normalizer",
                                    "n_train", "n_seeds", "masking_seed", "background_seed", "reference_seed"])
def test_altered_summary_or_procedure_metadata_is_rejected(completed, tmp_path, change):
    output, corpus = completed
    reports = tmp_path / "reports"
    shutil.copytree(output, reports)
    path = reports / "positive_control.json"
    payload = json.loads(path.read_text())
    task = payload["budget_sweep"][1]["tasks"][1]
    if change == "span":
        task["per_seed"][1]["span"] += 1.
    elif change == "decision":
        task["per_seed"][1]["fires"] = not task["per_seed"][1]["fires"]
    elif change == "contrast":
        payload["budget_sweep"][1]["contrast"]["n_defined"] += 1
    elif change == "primary_alias":
        payload["tasks"][0]["return"] += 1.
    elif change == "normalizer":
        payload["normalizer"]["mean"][0] += 1.
    elif change == "n_train":
        payload["n_train"] += 1
    elif change == "n_seeds":
        task["n_seeds"] += 1
    elif change == "masking_seed":
        task["per_seed"][1]["masking_seed"] += 1
    elif change == "background_seed":
        task["background_seed"] += 1
    else:
        task["reference_policy_seeds"][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="objective|reconstruction differs"):
        verify(reports, corpus)


def test_content_hash_and_corpus_identity_are_enforced(completed, tmp_path):
    output, corpus = completed
    reports = tmp_path / "reports"
    shutil.copytree(output, reports)
    archive = reports / "positive_control_returns.npz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive hash differs"):
        verify(reports, corpus)
    other = tmp_path / "other.npz"
    other.write_bytes(corpus.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="corpus hash differs"):
        verify(output, other)


def test_selected_release_cannot_be_relabelled_as_having_both_endpoints():
    with pytest.raises(ValueError, match="requires schema 2"):
        verify(ROOT / "data/releases/research_v1/reports", ROOT / "data/releases/market_snapshot_v1/corpus.npz")


def test_existing_evidence_is_refused_before_training(completed):
    output, corpus = completed
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    result = subprocess.run([sys.executable, str(SCRIPT), "--corpus", str(corpus), "--out", str(output)],
                            env=ENV, capture_output=True, text=True)
    assert result.returncode != 0
    assert "refusing existing" in result.stderr
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_corpus_changed_during_execution_cannot_be_labelled_as_original_evidence(completed, tmp_path, monkeypatch):
    reports, source = completed
    corpus, output = tmp_path / "corpus.npz", tmp_path / "reports"
    shutil.copyfile(source, corpus)
    original = json.loads((reports / "positive_control.json").read_text())["budget_sweep"][0]["tasks"]
    with np.load(reports / "positive_control_returns.npz") as saved:
        raw = {key: saved[key].copy() for key in
               ("returns", "masked_returns", "reference_returns", "reference_masked_returns")}
    def changed(name, *args):
        corpus.write_bytes(corpus.read_bytes() + b"changed during execution")
        task = ["prediction", "trading"].index(name)
        return copy.deepcopy(original[task]), {key: value[0, task] for key, value in raw.items()}
    monkeypatch.setattr(producer, "run_task", changed)
    monkeypatch.setenv("ALLOW_UNRESOLVED_NULLS", "1")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--corpus", str(corpus), "--out", str(output),
                                   "--n-null", "3", "--seeds", "2", "--budgets", "1", "--seed", "17"])
    with pytest.raises(SystemExit, match="corpus changed during execution"):
        producer.main()
    assert not (output / "positive_control.json").exists()
    assert not (output / "positive_control_returns.npz").exists()


def test_producer_retains_undefined_zero_scale_and_both_raw_endpoints(completed, monkeypatch):
    _, corpus = completed
    split = producer.walk_forward_split(producer.EpisodeBatch.load(str(corpus)))
    size = len(split.test)
    samples = iter([{"span": 0., "full": np.full(size, 2.), "masked": np.full(size, 2.)}] * 2
                   + [{"span": 1., "full": np.full(size, 2.), "masked": np.full(size, 1.)}])
    monkeypatch.setattr(producer, "train", lambda *args: None)
    monkeypatch.setattr(producer, "masked_span", lambda *args, **kwargs: next(samples))
    make_env = lambda batch: producer.BinaryMarketEnv(batch, normalizer=split.normalizer)
    make_roll = lambda batch: producer.VectorizedRollout(batch, normalizer=split.normalizer)
    summary, raw = producer.run_task("trading", make_env, make_roll, split, 1, 1, 2, 17)
    assert summary["result"]["z_score"] is summary["result"]["p_normal"] is None
    assert summary["verdict"] == "unresolved" and summary["fires"] is False
    np.testing.assert_array_equal(raw["masked_returns"], np.ones((1, size)))
    np.testing.assert_array_equal(raw["reference_masked_returns"], np.full((2, size), 2.))
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("args", [["--budgets", "1", "1"], ["--n-null", "1"], ["--seeds", "0"]])
def test_invalid_budgets_fail_before_training(tmp_path, args):
    output = tmp_path / "reports"
    result = subprocess.run([sys.executable, str(SCRIPT), "--corpus", str(tmp_path / "missing.npz"),
                             "--out", str(output), *args], env=ENV, capture_output=True, text=True)
    assert result.returncode != 0
    assert "require positive" in result.stderr
    assert not output.exists()
