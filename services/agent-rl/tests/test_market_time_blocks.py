"""Repeated episodes in one day must not manufacture independent evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_manifest import validate_evaluation, validate_stability_returns
from stability import compare_stability
from nano_rl.metrics import paired_bootstrap_p_value, cluster_bootstrap_ci, cluster_bootstrap_p_value


def data():
    a = np.repeat([1., -.5], 100)
    return a, np.zeros_like(a), np.repeat([0, 1], 100)


def test_day_blocks_preserve_shared_noise_that_episode_bootstrap_misses():
    a, b, days = data()
    assert paired_bootstrap_p_value(a, b) < .01
    probability = paired_bootstrap_p_value(a, b, block_ids=days)
    assert .48 < probability < .52
    assert probability == paired_bootstrap_p_value(np.array([1., -.5]), np.zeros(2), block_ids=[0, 1])


@pytest.mark.parametrize("blocks", [[0] * 200, [0, 1]])
def test_invalid_or_single_block_inference_is_rejected(blocks):
    a, b, _ = data()
    with pytest.raises(ValueError):
        paired_bootstrap_p_value(a, b, block_ids=blocks)


@pytest.mark.parametrize("behaviour,outcomes", [(np.nan, 1.), (1., np.nan), (np.nan, np.nan)])
def test_undefined_rank_comparison_cannot_report_lower_stability(behaviour, outcomes):
    assert compare_stability(behaviour, outcomes) == "undefined rank comparison"


def fixture(tmp_path):
    pnl = np.stack([np.repeat([1., -.5], 4), np.zeros(8)])
    epochs = np.r_[np.arange(4) * 900., 86400. + np.arange(4) * 900.]
    days = np.floor(epochs / 86400).astype(int)
    probability = paired_bootstrap_p_value(*pnl, 128, 3, block_ids=days)
    np.savez(tmp_path / "stability_returns.npz", returns=pnl, open_epoch=epochs)
    stability = {"n_seeds": 2, "seed_pnl": pnl.mean(axis=1).tolist(),
                 "performance_returns_file": "stability_returns.npz", "performance_pairs": [[0, 1]],
                 "performance_pvalues": [probability], "performance_bootstrap_resamples": 128,
                 "performance_bootstrap_seed": 3, "performance_utc_day_blocks": 2,
                 "n_pairs_without_detected_difference": int(probability >= .05)}
    (tmp_path / "stability.json").write_text(json.dumps(stability))
    zeros = [np.zeros(8)] * 2
    np.savez(tmp_path / "evaluation_returns.npz", ppo_pnl=pnl, open_epoch=epochs,
             **{"baseline_always-flat_pnl": zeros[0], "baseline_toy_pnl": pnl[0]})
    evaluation = {"test_episodes": 8, "ppo_across_seeds": {"mean_pnl_mean": float(pnl.mean()),
                  "mean_pnl_std": float(pnl.mean(axis=1).std())},
                  "ppo_per_seed": [{"mean_pnl": float(row.mean())} for row in pnl],
                  "paired_bootstrap_design": {"n_boot": 128, "seed": 3},
                  "paired_bootstrap_p_vs_flat": {"toy": probability},
                  "ppo_seed_day_bootstrap": {
                      "p_value": cluster_bootstrap_p_value(list(pnl), zeros, block_ids=days),
                      "ci95": list(cluster_bootstrap_ci(list(pnl), zeros, block_ids=days))}}
    (tmp_path / "results.json").write_text(json.dumps(evaluation))
    return pnl, epochs, stability, evaluation


@pytest.mark.parametrize("changed", ["return", "probability", "count"])
def test_stability_raw_evidence_rejects_altered_comparisons(tmp_path, changed):
    pnl, epochs, payload, _ = fixture(tmp_path)
    validate_stability_returns(tmp_path)
    if changed == "return":
        pnl[0, 0] += 1
        np.savez(tmp_path / "stability_returns.npz", returns=pnl, open_epoch=epochs)
    elif changed == "probability":
        payload["performance_pvalues"][0] += .1
    else:
        payload["n_pairs_without_detected_difference"] = 0
    (tmp_path / "stability.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_stability_returns(tmp_path)


@pytest.mark.parametrize("changed", ["probability", "interval"])
def test_market_table_and_primary_interval_reconstruct_from_returns(tmp_path, changed):
    _, _, _, payload = fixture(tmp_path)
    validate_evaluation(tmp_path)
    if changed == "probability":
        payload["paired_bootstrap_p_vs_flat"]["toy"] += .1
    else:
        payload["ppo_seed_day_bootstrap"]["ci95"][0] += 10
    (tmp_path / "results.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_evaluation(tmp_path)
