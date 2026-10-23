"""Manuscript summaries preserve pairing, raw evidence and undefined scales."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from positive_control import direct_contrast
from publication_statistics import nullable_z, seed_summary, state_statistics, steering_statistics


@pytest.mark.parametrize("z", [0., float("inf"), -float("inf")])
def test_zero_reference_scale_cannot_define_a_standardized_contrast(z):
    assert nullable_z({"z_score": z, "null_std": 0.}) is None
    a = {"per_seed": [{"seed": 0, "result": {"z_score": z, "null_std": 0.}}]}
    b = {"per_seed": [{"seed": 0, "result": {"z_score": 1., "null_std": 1.}}]}
    contrast = direct_contrast(a, b)
    assert contrast["per_seed"][0]["difference"] is None
    assert contrast["n_defined"] == 0
    assert contrast["mean_difference"] is None
    json.dumps(contrast, allow_nan=False)


def test_seed_spread_is_a_labeled_sample_sd_and_not_episode_spread():
    values = np.array([[1., 3.], [3., 5.]])
    result = seed_summary(values)
    assert result["mean"] == 3.
    assert result["seed_means"] == [2., 4.]
    assert result["seed_sd"] == pytest.approx(np.sqrt(2))
    assert result["seed_sd_ddof"] == 1
    assert seed_summary(values[:1])["seed_sd"] is None


def test_state_intervals_reconstruct_constant_paired_effect_and_reject_tampering(tmp_path):
    original = np.array([[2., 3., 4., 5.], [3., 4., 5., 6.]])
    epochs = np.array([0., 900., 86400., 87300.])
    modes = {"full_observation": 2., "market_rows": 1., "market_trajectories": 3.}
    raw = {"market__unmodified": original, "market__open_epoch": epochs}
    raw.update({f"market__{mode}": original - effect for mode, effect in modes.items()})
    np.savez(tmp_path / "state_masking_returns.npz", **raw)
    payload = {"limitations": ["fixed masking draws"], "cases": [{
        "case": "market", "n_episodes": 4,
        "per_seed": [{"unmodified_return": float(row.mean()), "spans": modes}
                     for row in original],
        "interventions": {mode: {"mean_span": effect, "per_seed_spans": [effect] * 2,
                                  "ci95": [effect] * 2} for mode, effect in modes.items()},
    }]}
    path = tmp_path / "state_masking.json"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    result = state_statistics(tmp_path, market_returns=original, market_epochs=epochs)
    assert result["cases"][0]["interventions"]["market_rows"]["ci95"] == [1., 1.]
    assert result["cases"][0]["n_resampling_blocks"] == 2
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="unmodified market returns"):
        state_statistics(tmp_path, market_returns=original + 1, market_epochs=epochs)
    payload["cases"][0]["interventions"]["market_rows"]["ci95"] = [0., 0.]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="market/market_rows interval"):
        state_statistics(tmp_path)


def test_steering_uses_paired_seed_and_shared_block_effects_without_rewriting_inputs(tmp_path):
    baseline = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]])
    epochs = [0., 900., 86400., 87300.]
    rows = []
    for penalty, values in ((0., baseline), (20., baseline + 1)):
        row = {"coef": penalty, "pnl": values.tolist(), "return_mean": float(values.mean()),
               "return_per_seed": values.mean(axis=1).tolist(),
               "target_share_per_seed": [.2, .4], "target_share_mean": .3}
        if penalty:
            row.update(return_diff=1., return_diff_ci=[1., 1.], p_cluster=1 / 10001)
        rows.append(row)
    payload = {"real_market": rows, "learnable_synthetic": rows,
               "evaluation_open_epoch": {"market": epochs, "synthetic": epochs},
               "resampling": {"market": "shared days", "synthetic": "shared episodes"}}
    path = tmp_path / "steering.json"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    result = steering_statistics(tmp_path)
    assert result["reported_penalty"] == 20.
    assert result["populations"]["real_market"][1]["return_difference"] == 1.
    assert result["populations"]["real_market"][1]["return_diff_ci95"] == [1., 1.]
    assert path.read_bytes() == before
    payload["real_market"][1]["p_cluster"] = 1.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="paired probability"):
        steering_statistics(tmp_path)
