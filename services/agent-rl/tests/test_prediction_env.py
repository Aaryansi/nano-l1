"""tests for the prediction task used as the real-data positive control.

the parity test is the important one. PredictionRollout exists because
attribution needs a batched replay, and it duplicates PredictionEnv's reward
logic. if the two ever disagree, every attribution measured on the prediction
task is measured against a different task from the one the agent trained on,
and nothing would visibly break. the trading env has the same guarantee in
tests/test_rollout_parity.py; this is its counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.env.features import N_FEATURES, fit_normalizer
from nano_rl.env.prediction import (
    BlindEnv,
    PredictionEnv,
    PredictionRollout,
    observation_moments,
    outcome_sign,
)
from nano_rl.env.synthetic import make_null_corpus


@pytest.fixture(scope="module")
def corpus():
    b = make_null_corpus(n_episodes=24, n_steps=6, seed=0)
    return b, fit_normalizer(b)


class TestReward:
    def test_correct_call_pays_one_per_step(self, corpus):
        batch, norm = corpus
        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        env.reset(seed=0, options={"episode": 0})
        settled = float(batch.settlement[0])
        # action 2 is a YES call, action 0 is NO
        good = 2 if settled > 0.5 else 0
        total = 0.0
        while True:
            _, r, term, _, _ = env.step(good)
            total += r
            if term:
                break
        assert total == pytest.approx(batch.n_steps)

    def test_wrong_call_is_symmetric(self, corpus):
        batch, norm = corpus
        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        env.reset(seed=0, options={"episode": 0})
        settled = float(batch.settlement[0])
        bad = 0 if settled > 0.5 else 2
        total = sum(env.step(bad)[1] for _ in range(batch.n_steps))
        assert total == pytest.approx(-batch.n_steps)

    def test_abstaining_scores_zero(self, corpus):
        batch, norm = corpus
        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        env.reset(seed=0, options={"episode": 0})
        total = sum(env.step(1)[1] for _ in range(batch.n_steps))
        assert total == 0.0

    def test_no_fees_are_charged(self, corpus):
        """the prediction task has no execution, so cost must never appear."""
        batch, norm = corpus
        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        env.reset(seed=0, options={"episode": 0})
        # alternating calls would be expensive under the trading objective
        rewards = [env.step(k % 3)[1] for k in range(batch.n_steps)]
        assert all(abs(r) in (0.0, 1.0) for r in rewards)

    def test_outcome_sign_maps_to_plus_minus_one(self):
        np.testing.assert_array_equal(
            outcome_sign(np.array([0.0, 1.0])), np.array([-1.0, 1.0])
        )


class TestRolloutParity:
    @pytest.mark.parametrize("action", [0, 1, 2])
    def test_constant_policy_matches_the_env(self, corpus, action):
        batch, norm = corpus
        roll = PredictionRollout(batch, normalizer=norm, max_position=100.0)
        got = roll.run(lambda obs: np.full(len(obs), action))["returns"]

        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        want = []
        for ep in range(len(batch)):
            env.reset(seed=0, options={"episode": ep})
            want.append(sum(env.step(action)[1] for _ in range(batch.n_steps)))
        np.testing.assert_allclose(got, np.asarray(want), atol=1e-9)

    def test_observations_have_the_shared_feature_layout(self, corpus):
        batch, norm = corpus
        roll = PredictionRollout(batch, normalizer=norm, max_position=100.0)
        z = np.zeros(len(batch))
        assert roll.observations(0, z, z, z).shape == (len(batch), N_FEATURES)


class TestBlindEnv:
    def test_observation_ignores_the_underlying_state(self, corpus):
        batch, norm = corpus
        inner = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        mean, sd = observation_moments(inner, n_steps=200, seed=0)

        a = BlindEnv(PredictionEnv(batch, normalizer=norm, max_position=100.0),
                     mean, sd, seed=7)
        b = BlindEnv(PredictionEnv(batch, normalizer=norm, max_position=100.0),
                     mean, sd, seed=7)
        # same rng seed, different episodes: observations must still coincide,
        # which they can only do if the real state is not being read.
        oa, _ = a.reset(seed=0, options={"episode": 0})
        ob, _ = b.reset(seed=0, options={"episode": 3})
        np.testing.assert_allclose(oa, ob)

    def test_reward_is_untouched_by_blinding(self, corpus):
        batch, norm = corpus
        inner = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        mean, sd = observation_moments(inner, n_steps=200, seed=0)

        plain = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        blind = BlindEnv(PredictionEnv(batch, normalizer=norm, max_position=100.0),
                         mean, sd, seed=1)
        plain.reset(seed=0, options={"episode": 2})
        blind.reset(seed=0, options={"episode": 2})
        for _ in range(batch.n_steps):
            assert plain.step(2)[1] == blind.step(2)[1]

    def test_forwards_attributes_the_agent_needs(self, corpus):
        batch, norm = corpus
        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        mean, sd = observation_moments(env, n_steps=100, seed=0)
        blind = BlindEnv(env, mean, sd, seed=0)
        # gymnasium 1.0 stopped forwarding these through wrappers
        assert blind.batch is batch
        assert blind.action_space.n == 3
        assert blind.observation_space.shape == (N_FEATURES,)

    def test_moments_have_one_entry_per_feature(self, corpus):
        batch, norm = corpus
        env = PredictionEnv(batch, normalizer=norm, max_position=100.0)
        mean, sd = observation_moments(env, n_steps=200, seed=0)
        assert mean.shape == (N_FEATURES,)
        assert sd.shape == (N_FEATURES,)
        assert (sd > 0).all()
