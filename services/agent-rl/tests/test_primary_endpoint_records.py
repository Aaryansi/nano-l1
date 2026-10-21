"""Raw episode evidence must reproduce every primary decision, including all seeds."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_manifest import validate_primary_endpoints
from calibrate_detection import summarize
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.env.features import fit_normalizer
from nano_rl.env.synthetic import make_learnable_corpus
from nano_rl.explain.rollout import VectorizedRollout, build_background, masked_span
from nano_rl.explain.sanity import test_span_against_null as compare
from sanity_check_explanations import endpoint_span


def test_endpoint_recording_preserves_span_and_randomness():
    batch = make_learnable_corpus(n_episodes=32, n_steps=6, seed=42)
    norm = fit_normalizer(batch)
    agent = PPOAgent(18, 3, PPOConfig(seed=12))
    roll = VectorizedRollout(batch, normalizer=norm)
    bg = build_background(roll, n_samples=256, seed=7)
    expected = masked_span(agent, roll, bg, 7)
    raw = {}
    assert endpoint_span(agent, batch, norm, 7, raw=raw, key="test") == expected
    assert float(raw["test_full"].mean() - raw["test_masked"].mean()) == expected
    np.testing.assert_array_equal(raw["test_open_epoch"], batch.open_epoch)
    with pytest.raises(ValueError, match="duplicate"):
        endpoint_span(agent, batch, norm, 7, raw=raw, key="test")


def fixture(tmp_path):
    n = 8
    raw = {}
    spans = {"reference_0": -1., "reference_1": 1., "signal": 4.,
             "null": 0., "market_0": 2., "market_1": 3.}
    for key, span in spans.items():
        raw[f"{key}_full"] = np.arange(n, dtype=float) + span
        raw[f"{key}_masked"] = np.arange(n, dtype=float)
        raw[f"{key}_open_epoch"] = np.arange(n, dtype=float) * 900
    np.savez(tmp_path / "endpoints.npz", **raw)
    nulls = [-1., 1.]
    result = lambda key: compare(spans[key], nulls).as_dict()
    payload = {"endpoint_returns_file": "endpoints.npz", "synthetic_episodes": n,
               "endpoint_evaluation_episodes": n, "null_spans": nulls,
               "reference_endpoint_keys": ["reference_0", "reference_1"],
               "observed_endpoint_keys": {"planted_signal": "signal", "held_out_null": "null", "real_market": "market_0"},
               "results": {"planted_signal": result("signal"), "held_out_null": result("null"), "real_market": result("market_0")},
               "real_endpoint_keys": ["market_0", "market_1"], "real_spans_per_seed": [2., 3.],
               "real_results_per_seed": [result("market_0"), result("market_1")]}
    (tmp_path / "sanity_test.json").write_text(json.dumps(payload))
    row = {"endpoint_key": "signal", "data": {"evaluation_episodes": n},
           "return": 7.5, "result": result("signal")}
    case = {"signal_strength": 1., "trials": [row], **summarize([row])}
    calibration = {"endpoint_returns_file": "endpoints.npz", "synthetic_episodes": n,
                   "reference_endpoint_keys": payload["reference_endpoint_keys"],
                   "null_spans": nulls, "cases": [case]}
    (tmp_path / "detection_calibration.json").write_text(json.dumps(calibration))
    return raw, payload, calibration


@pytest.mark.parametrize("changed", ["signal_masked", "reference_1_full", "market_1_masked", "null_open_epoch"])
def test_validation_catches_modified_raw_endpoints(tmp_path, changed):
    raw, _, _ = fixture(tmp_path)
    validate_primary_endpoints(tmp_path)
    raw[changed][0] += 900 if changed.endswith("open_epoch") else 1
    np.savez(tmp_path / "endpoints.npz", **raw)
    with pytest.raises(ValueError):
        validate_primary_endpoints(tmp_path)


@pytest.mark.parametrize("changed", ["decision", "rate"])
def test_validation_catches_modified_calibration_claims(tmp_path, changed):
    _, _, payload = fixture(tmp_path)
    validate_primary_endpoints(tmp_path)
    if changed == "decision":
        payload["cases"][0]["trials"][0]["result"]["passes"] = True
    else:
        payload["cases"][0]["n_rejected"] = 1
    (tmp_path / "detection_calibration.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_primary_endpoints(tmp_path)
