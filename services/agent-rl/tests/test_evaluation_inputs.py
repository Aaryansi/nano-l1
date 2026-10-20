"""An isolated evaluation must not discover unrelated local model artifacts."""
import importlib.util
from pathlib import Path

import pytest

from nano_rl.baselines import default_baselines

spec = importlib.util.spec_from_file_location("evaluate_inputs", Path(__file__).resolve().parents[1] / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_optional_models_are_absent_even_when_legacy_files_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    primary = tmp_path / "fresh"
    primary.mkdir()
    (primary / "seed0.pt").touch()
    legacy = tmp_path / "runs/ppo_nofric"
    legacy.mkdir(parents=True)
    (legacy / "seed0.pt").touch()
    args = evaluation.parse_args(["--corpus", "corpus.npz", "--runs", str(primary)])
    assert args.runs_nofric is None
    assert args.logistic_model is None
    assert "momentum-fallback" in [policy.name for policy in default_baselines()]
    assert "logistic-shipped" not in [policy.name for policy in default_baselines()]


def test_explicit_missing_comparison_checkpoints_are_an_error(tmp_path):
    primary = tmp_path / "fresh"
    primary.mkdir()
    (primary / "seed0.pt").touch()
    with pytest.raises(SystemExit) as error:
        evaluation.parse_args(["--corpus", "corpus.npz", "--runs", str(primary),
                               "--runs-nofric", str(tmp_path / "missing")])
    assert error.value.code == 2
