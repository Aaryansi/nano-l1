"""Unresolved references cannot establish agreement or decision retention."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts"))
from bootstrap_z import bootstrap
from run_manifest import validate_reference_bootstraps


def test_unresolved_bootstrap_point_has_no_resolved_decision_retention():
    result = bootstrap(10., [0., 1., 2., 3.], 100, 4)
    assert result["verdict"] == "unresolved"
    assert result["verdict_stability"] is None
    assert result["verdict_counts"]["unresolved"] == 100
    assert sum(result["verdict_counts"].values()) == 100
    json.dumps(result, allow_nan=False)


def test_degenerate_resamples_do_not_retain_a_resolved_nonrejection():
    nulls = np.array([0.] * 19 + [1.])
    result = bootstrap(0., nulls, 500, 7)
    # Some resamples contain only zeros. The previous implementation dropped
    # these and reported 100% retention of a resolved non-rejection.
    draws = nulls[np.random.default_rng(7).integers(0, 20, size=(500, 20))]
    n_degenerate = int(np.sum(draws.std(axis=1, ddof=1) <= 1e-12))
    assert 0 < n_degenerate < 500
    assert result["verdict"] == "not distinguishable from null"
    assert result["verdict_counts"]["unresolved"] == n_degenerate
    assert result["verdict_counts"]["informative"] == 0
    assert result["verdict_stability"] == (500 - n_degenerate) / 500
    assert result["n_defined_z"] == 500 - n_degenerate


def test_fully_degenerate_bootstrap_serializes_undefined_diagnostics():
    result = bootstrap(1., [0.] * 24, 20, 0)
    assert result["z"] is None
    assert result["z_lo"] is None and result["z_hi"] is None
    assert result["n_defined_z"] == 0
    assert result["verdict_stability"] is None
    assert result["verdict_counts"]["unresolved"] == 20
    json.dumps(result, allow_nan=False)


def test_zero_bootstrap_budget_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        bootstrap(0., [0., 1.], 0, 0)


def test_all_unresolved_bootstrap_cli_and_validation(tmp_path):
    source = {"observed_span": 1., "null_synthetic": {"spans": [0.] * 24},
              "null_blinded_real": {"spans": [0.] * 24}}
    (tmp_path / "null_corpus_check.json").write_text(json.dumps(source))
    result = subprocess.run([sys.executable, str(RL / "scripts/bootstrap_z.py"),
                             "--reports", str(tmp_path), "--out", str(tmp_path),
                             "--n-boot", "20"], cwd=RL,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    path = tmp_path / "z_intervals.json"
    payload = json.loads(path.read_text(), parse_constant=lambda token: pytest.fail(token))
    assert all(row["verdict_stability"] is None for row in payload.values())
    validate_reference_bootstraps(tmp_path)
    next(iter(payload.values()))["verdict_stability"] = 1.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reference resampling summary differs"):
        validate_reference_bootstraps(tmp_path)
