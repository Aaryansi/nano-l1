"""Shared evaluation noise must survive resampling across training seeds."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nano_rl.metrics import cluster_bootstrap_ci, cluster_bootstrap_p_value
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.env.synthetic import make_null_corpus
from nano_rl.env.features import fit_normalizer
from nano_rl.env.binary_market import BinaryMarketEnv


def test_duplicating_seed_rows_does_not_remove_shared_episode_uncertainty():
    noise = np.random.default_rng(4).normal(0, 10, 200)
    widths = []
    for k in (2, 20):
        lo, hi = cluster_bootstrap_ci([noise] * k, [np.zeros(200)] * k,
                                      n_boot=2000, seed=31)
        widths.append(hi - lo)
    assert .8 < widths[1] / widths[0] < 1.2


def test_day_blocks_preserve_common_within_day_shocks():
    shocks = np.repeat(np.array([-10., 10., -8., 8., -6., 6., -4., 4.]), 20)
    a, b = [shocks] * 3, [np.zeros(160)] * 3
    lo, hi = cluster_bootstrap_ci(a, b, n_boot=2000)
    blo, bhi = cluster_bootstrap_ci(a, b, n_boot=2000, block_ids=np.repeat(np.arange(8), 20))
    assert bhi - blo > 2 * (hi - lo)


def test_misaligned_seed_pairs_are_not_silently_truncated():
    with pytest.raises(ValueError, match="equal counts"):
        cluster_bootstrap_p_value([np.ones(4)] * 3, [np.zeros(4)] * 2)


def test_critic_calibrates_flat_policy_return_in_dollars():
    spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).resolve().parents[1] / "scripts/evaluate.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    batch = make_null_corpus(n_episodes=32, n_steps=6, seed=2)
    env = BinaryMarketEnv(batch, normalizer=fit_normalizer(batch))
    agent = PPOAgent(18, 3, PPOConfig(seed=3))
    with torch.no_grad():
        for parameter in agent.net.parameters():
            parameter.zero_()
        agent.net.policy_head.bias[1] = 1000.
    result = evaluation.critic_calibration(agent, env, repetitions=2)
    np.testing.assert_array_equal(result["predictions"], np.zeros(32))
    np.testing.assert_array_equal(result["realized_returns"], np.zeros(32))
    assert all(a == b == 0. for a, b, _ in result["bins"])
