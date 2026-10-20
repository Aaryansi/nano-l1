"""State-preserving replacements obey their declared input invariants."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nano_rl.agents.networks import ActorCritic
from nano_rl.env.features import MARKET_FEATURES, N_FEATURES, SIGNAL_OBS_IDX, feature_index, fit_normalizer
from nano_rl.env.synthetic import make_null_corpus
from nano_rl.explain.interventions import intervention_rollout
from nano_rl.explain.rollout import VectorizedRollout, build_background, greedy_policy, masked_span


@pytest.mark.parametrize("mode", ["market_rows", "market_trajectories"])
def test_market_replacement_preserves_actual_inventory_and_clock(mode):
    batch = make_null_corpus(n_episodes=32, n_steps=6, seed=3)
    roll = VectorizedRollout(batch, normalizer=fit_normalizer(batch))
    background = build_background(roll)
    original, replaced = [], []

    def record(rows):
        def policy(obs):
            rows.append(obs.copy())
            return np.where(abs(obs[:, feature_index("position")]) < .5, 2, 1)
        return policy

    expected = roll.run(record(original))
    actual = intervention_rollout(record(replaced), roll, background, mode, seed=5)
    np.testing.assert_array_equal(actual["returns"], expected["returns"])
    keep = [feature_index("time_to_expiry_frac"), *range(len(MARKET_FEATURES), N_FEATURES)]
    np.testing.assert_array_equal(np.array(original)[:, :, keep], np.array(replaced)[:, :, keep])
    assert not np.array_equal(np.array(original)[:, :, SIGNAL_OBS_IDX],
                              np.array(replaced)[:, :, SIGNAL_OBS_IDX])
    full = intervention_rollout(record([]), roll, background, "full_observation", seed=5)
    assert not np.array_equal(full["returns"], expected["returns"])


def test_donor_market_trajectory_stays_with_one_episode():
    batch = make_null_corpus(n_episodes=32, n_steps=6, seed=8)
    norm = fit_normalizer(batch)
    roll = VectorizedRollout(batch, normalizer=norm)
    seen = []

    def policy(obs):
        seen.append(obs.copy())
        return np.ones(len(obs), dtype=int)

    intervention_rollout(policy, roll, build_background(roll), "market_trajectories", seed=4)
    columns = [i for i in range(len(MARKET_FEATURES)) if i != feature_index("time_to_expiry_frac")]
    market = norm.transform(roll.market.reshape(-1, len(MARKET_FEATURES))).reshape(roll.market.shape)
    for recipient in range(len(batch)):
        coherent = [all(np.allclose(seen[t][recipient, columns], market[donor, t, columns])
                        for t in range(batch.n_steps)) for donor in range(len(batch))]
        assert any(coherent)


def test_full_replacement_matches_existing_headline_endpoint():
    batch = make_null_corpus(n_episodes=32, seed=9)
    roll = VectorizedRollout(batch, normalizer=fit_normalizer(batch))
    background = build_background(roll, seed=1)
    torch.manual_seed(6)
    agent = SimpleNamespace(net=ActorCritic(N_FEATURES, 3, 64))
    base = greedy_policy(agent)
    expected = masked_span(agent, roll, background, seed=3)
    actual = (roll.run(base)["returns"].mean() - intervention_rollout(
        base, roll, background, "full_observation", seed=3)["returns"].mean())
    assert actual == pytest.approx(expected, abs=1e-12)
