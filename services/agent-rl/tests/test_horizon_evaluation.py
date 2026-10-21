"""Trained reference policies must retain their preprocessing at evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import horizon_scaling as horizon
from run_manifest import validate_horizon_returns


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
    assert len(checked) == 2
    assert row["shared_data"]["train_seed"] != row["shared_data"]["evaluation_seed"]
    assert row["shared_data"]["evaluation_seed"] not in row["reference_train_seeds"]
    assert row["reference_normalizers"][0] != row["weight_normalizer"]
    np.savez(tmp_path / "endpoints.npz", **raw)
    payload = {"rows": [row], "evaluation_episodes": 16, "endpoint_returns_file": "endpoints.npz"}
    (tmp_path / "horizon_scaling.json").write_text(json.dumps(payload))
    validate_horizon_returns(tmp_path)
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
    row["ratio"] = 1e9
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="width ratio differs"):
        validate_horizon_returns(tmp_path)
