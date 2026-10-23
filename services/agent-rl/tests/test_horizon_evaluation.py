"""Trained reference policies must retain their preprocessing at evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import horizon_scaling as horizon
from run_manifest import validate_horizon_returns
from portable_validation import validate_horizon_returns as portable_horizons


def test_horizon_uses_each_training_normalizer_and_saves_reconstructable_endpoints(monkeypatch, tmp_path):
    trained = {}
    checked = []
    original_train, original_span = horizon.PPOAgent.train, horizon.span_of

    def train(self, env, **kwargs):
        trained[id(self.net)] = env.normalizer
        return original_train(self, env, **kwargs)

    def span(net, batch, norm, *args):
        if id(net) in trained:
            assert norm is trained[id(net)]
            checked.append(id(net))
        return original_span(net, batch, norm, *args)

    monkeypatch.setattr(horizon.PPOAgent, "train", train)
    monkeypatch.setattr(horizon, "span_of", span)
    raw = {}
    row = horizon.measure_horizon(4, 2, 1, 16, 0, raw=raw)
    assert row["clock_feature_deadline_seconds"] == 300
    assert len(checked) == 2
    assert row["shared_data"]["train_seed"] != row["shared_data"]["evaluation_seed"]
    assert row["shared_data"]["evaluation_seed"] not in row["reference_train_seeds"]
    assert row["reference_normalizers"][0] != row["weight_normalizer"]
    np.savez(tmp_path / "endpoints.npz", **raw)
    payload = {"rows": [row], "evaluation_episodes": 16, "endpoint_returns_file": "endpoints.npz"}
    (tmp_path / "horizon_scaling.json").write_text(json.dumps(payload))
    validate_horizon_returns(tmp_path)
    portable_horizons(tmp_path)
    row["clock_feature_deadline_seconds"] = 900
    (tmp_path / "horizon_scaling.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="horizon clock deadline differs"):
        validate_horizon_returns(tmp_path)
    with pytest.raises(ValueError, match="horizon clock deadline"):
        portable_horizons(tmp_path)
    row["clock_feature_deadline_seconds"] = 300
    payload["alpha_weight"] = 100.
    (tmp_path / "horizon_scaling.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="horizon fit differs"):
        validate_horizon_returns(tmp_path)
    payload.pop("alpha_weight")
    (tmp_path / "horizon_scaling.json").write_text(json.dumps(payload))
    key = row["endpoint_keys"]["env"][0]
    raw[f"{key}_masked"][0] += 1
    np.savez(tmp_path / "endpoints.npz", **raw)
    with pytest.raises(ValueError, match="differs from raw"):
        validate_horizon_returns(tmp_path)
    with pytest.raises(ValueError, match="horizon env spans"):
        portable_horizons(tmp_path)


def test_horizon_fit_does_not_manufacture_a_log_scale_for_zero_width():
    assert horizon.fit_power_law(np.array([1., 2., 3.]), np.zeros(3)) == (None, None)
    slope, quality = horizon.fit_power_law(np.array([1., 2., 3.]), np.array([1., 4., 9.]))
    assert slope == pytest.approx(2.)
    assert quality == pytest.approx(1.)


def test_zero_width_ratio_is_undefined_and_cannot_be_replaced_by_a_large_number(monkeypatch, tmp_path):
    monkeypatch.setattr(horizon.PPOAgent, "train", lambda *args, **kwargs: None)
    monkeypatch.setattr(horizon, "span_of", lambda net, batch, *args: {
        "span": 0., "full": np.zeros(len(batch)), "masked": np.zeros(len(batch))})
    raw = {}
    row = horizon.measure_horizon(4, 2, 1, 8, 0, raw=raw)
    assert row["ratio"] is None
    np.savez(tmp_path / "endpoints.npz", **raw)
    payload = {"rows": [row], "evaluation_episodes": 8, "endpoint_returns_file": "endpoints.npz"}
    path = tmp_path / "horizon_scaling.json"
    path.write_text(json.dumps(payload, allow_nan=False))
    validate_horizon_returns(tmp_path)
    portable_horizons(tmp_path)
    row["ratio"] = 1e9
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="width ratio differs"):
        validate_horizon_returns(tmp_path)
    with pytest.raises(ValueError, match="horizon width ratio"):
        portable_horizons(tmp_path)


def test_portable_horizon_fit_accepts_only_roundoff_in_reconstructed_slope(tmp_path):
    raw, rows = {}, []
    for length in (4, 8, 16):
        row = {"n_steps": length, "clock_feature_deadline_seconds": 60 * (length + 1),
               "endpoint_keys": {}}
        for arm, power in (("weight", 1), ("env", 2)):
            spans = [-float(length ** power), float(length ** power)]
            keys = [f"{arm}_{length}_{index}" for index in range(2)]
            row["endpoint_keys"][arm] = keys
            row[f"{arm}_spans"], row[f"{arm}_std"] = spans, float(np.std(spans, ddof=1))
            for key, value in zip(keys, spans, strict=True):
                raw[f"{key}_full"], raw[f"{key}_masked"] = np.full(4, value), np.zeros(4)
                raw[f"{key}_open_epoch"] = np.arange(4, dtype=float)
        row["ratio"] = row["weight_std"] / row["env_std"]
        rows.append(row)
    np.savez(tmp_path / "endpoints.npz", **raw)
    x = np.array([4., 8., 16.])
    slope, _ = horizon.fit_power_law(x, np.array([row["weight_std"] for row in rows]))
    payload = {"rows": rows, "evaluation_episodes": 4, "endpoint_returns_file": "endpoints.npz",
               "alpha_weight": float(np.nextafter(slope, float("inf")))}
    path = tmp_path / "horizon_scaling.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="horizon fit differs"):
        validate_horizon_returns(tmp_path)
    portable_horizons(tmp_path)
    payload["alpha_weight"] += .1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fit slope"):
        portable_horizons(tmp_path)
