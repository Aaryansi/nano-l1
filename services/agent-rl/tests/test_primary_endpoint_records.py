"""Raw episode evidence must reproduce every primary decision, including all seeds."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_manifest import validate_primary_endpoints
from portable_validation import (reconstructed, validate_primary_endpoints as portable_primary,
                                 validate_reference_bootstraps as portable_bootstraps)
from calibrate_detection import summarize
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.env.features import fit_normalizer
from nano_rl.env.synthetic import make_learnable_corpus
from nano_rl.explain.rollout import VectorizedRollout, build_background, masked_span
from nano_rl.explain.sanity import test_span_against_null as compare
from nano_rl.explain.rank import rank_comparisons
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
           "return": 7.5, "result": result("signal"), "rank_comparators": rank_comparisons(4., nulls)}
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
    with pytest.raises(ValueError):
        portable_primary(tmp_path)


@pytest.mark.parametrize("changed", ["decision", "rate", "rank"])
def test_validation_catches_modified_calibration_claims(tmp_path, changed):
    _, _, payload = fixture(tmp_path)
    validate_primary_endpoints(tmp_path)
    if changed == "decision":
        payload["cases"][0]["trials"][0]["result"]["passes"] = True
    elif changed == "rate":
        payload["cases"][0]["n_rejected"] = 1
    else:
        payload["cases"][0]["trials"][0]["rank_comparators"]["greater"]["p_value"] = .01
    (tmp_path / "detection_calibration.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_primary_endpoints(tmp_path)
    with pytest.raises(ValueError):
        portable_primary(tmp_path)


def test_portable_reader_accepts_normal_tail_roundoff_without_changing_evidence(tmp_path):
    _, payload, _ = fixture(tmp_path)
    row = payload["results"]["planted_signal"]
    row["p_normal"] = float(np.nextafter(row["p_normal"], 1.))
    path = tmp_path / "sanity_test.json"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="primary comparison"):
        validate_primary_endpoints(tmp_path)
    differences = []
    portable_primary(tmp_path, differences)
    assert path.read_bytes() == before
    assert any(context.endswith("p_normal") for _, context, _, _ in differences)


def test_portable_bootstrap_reader_permits_roundoff_but_rejects_changed_state_counts(tmp_path):
    from bootstrap_z import bootstrap, collect
    from run_manifest import validate_reference_bootstraps
    fixture(tmp_path)
    payload = {label: bootstrap(observed, nulls, 25, 0)
               for label, observed, nulls in collect(tmp_path)}
    first = next(iter(payload.values()))
    first["z_lo"] = float(np.nextafter(first["z_lo"], float("inf")))
    path = tmp_path / "z_intervals.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reference resampling"):
        validate_reference_bootstraps(tmp_path)
    portable_bootstraps(tmp_path)
    first["n_defined_z"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="n_defined_z"):
        portable_bootstraps(tmp_path)


@pytest.mark.parametrize("actual,recorded", [
    (1. + 1e-6, 1.),
    (1., float("inf")),
    (float("nan"), float("nan")),
    (1, True),
    (2., 2),
    ({"value": 1.}, {"value": 1., "count": 2}),
    ([1.], [1., 2.]),
    ("within", "unresolved"),
])
def test_portable_numeric_tolerance_never_changes_metadata_or_decisions(actual, recorded):
    with pytest.raises(ValueError, match="reconstruction differs"):
        reconstructed(actual, recorded, "test")
