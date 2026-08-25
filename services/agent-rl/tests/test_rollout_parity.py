"""the vectorised rollout must agree with the reference env, exactly.

nano_rl/explain/rollout.py reimplements the pnl accounting so that attribution
can replay hundreds of batches cheaply. two implementations of the same logic
is a genuine hazard: if they drift, the fast one produces confident and wrong
explanations, and nothing else in the project would notice.

so this file checks parity on returns, trades, fees, final position, and the
per-step observation vector, across every policy shape that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.costs import CostModel
from nano_rl.env.features import fit_normalizer
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus
from nano_rl.explain.rollout import VectorizedRollout, build_background

SHORT, FLAT, LONG = 0, 1, 2


def reference_run(batch, policy_fn, normalizer=None, max_position=100.0,
                  costs_enabled=True):
    """run the real env episode by episode, as the ground truth."""
    env = BinaryMarketEnv(
        batch,
        cost_model=CostModel(enabled=costs_enabled),
        normalizer=normalizer,
        max_position=max_position,
        random_episode_order=False,
    )
    rets, trades, fees, pos = [], [], [], []
    for ep in range(len(batch)):
        obs, _ = env.reset(options={"episode": ep})
        total, info, t = 0.0, {}, 0
        while True:
            a = policy_fn(obs[None, :], t)[0]
            obs, r, done, _, info = env.step(int(a))
            total += r
            t += 1
            if done:
                break
        rets.append(total)
        trades.append(info["trades"])
        fees.append(info["fees"])
        pos.append(info["position"])
    return {
        "returns": np.array(rets),
        "trades": np.array(trades),
        "fees": np.array(fees),
        "final_position": np.array(pos),
    }


@pytest.fixture
def corpus():
    return make_learnable_corpus(n_episodes=40, n_steps=8, seed=0)


CONSTANT_POLICIES = [
    ("always flat", FLAT),
    ("always long", LONG),
    ("always short", SHORT),
]


class TestConstantPolicies:
    @pytest.mark.parametrize("name,action", CONSTANT_POLICIES)
    def test_parity(self, corpus, name, action) -> None:
        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)

        got = vec.run(lambda obs: np.full(len(obs), action))
        want = reference_run(corpus, lambda obs, t: np.full(len(obs), action), norm)

        for key in ("returns", "trades", "fees", "final_position"):
            np.testing.assert_allclose(
                got[key], want[key], atol=1e-9, err_msg=f"{name}: {key} differs"
            )


class TestVaryingPolicies:
    def test_alternating_churn(self, corpus) -> None:
        """maximal churn exercises the fee and average-entry paths hardest."""
        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)

        # step parity decides the action, so both runners see the same sequence
        counter = {"t": 0}

        def vec_policy(obs):
            a = LONG if counter["t"] % 2 == 0 else SHORT
            counter["t"] += 1
            return np.full(len(obs), a)

        got = vec.run(vec_policy)
        want = reference_run(
            corpus, lambda obs, t: np.full(len(obs), LONG if t % 2 == 0 else SHORT), norm
        )
        for key in ("returns", "trades", "fees", "final_position"):
            np.testing.assert_allclose(got[key], want[key], atol=1e-9, err_msg=key)

    def test_position_flip_through_zero(self, corpus) -> None:
        """long, then short, then flat: covers opening, flipping, and closing."""
        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)
        seq = [LONG, LONG, SHORT, SHORT, FLAT, LONG, FLAT, SHORT]

        counter = {"t": 0}

        def vec_policy(obs):
            a = seq[counter["t"] % len(seq)]
            counter["t"] += 1
            return np.full(len(obs), a)

        got = vec.run(vec_policy)
        want = reference_run(
            corpus, lambda obs, t: np.full(len(obs), seq[t % len(seq)]), norm
        )
        for key in ("returns", "trades", "fees", "final_position"):
            np.testing.assert_allclose(got[key], want[key], atol=1e-9, err_msg=key)

    def test_observation_dependent_policy(self, corpus) -> None:
        """a policy that reads the observation, so obs parity is exercised too."""
        from nano_rl.env.features import SIGNAL_OBS_IDX

        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)

        def decide(obs):
            return np.where(obs[:, SIGNAL_OBS_IDX] > 0, LONG, SHORT)

        got = vec.run(decide)
        want = reference_run(corpus, lambda obs, t: decide(obs), norm)
        for key in ("returns", "trades", "fees", "final_position"):
            np.testing.assert_allclose(got[key], want[key], atol=1e-9, err_msg=key)


class TestObservationParity:
    def test_observations_match_step_by_step(self, corpus) -> None:
        """the observation vectors themselves must be identical, not just pnl."""
        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)
        env = BinaryMarketEnv(
            corpus, normalizer=norm, max_position=100.0, random_episode_order=False
        )

        n = len(corpus)
        pos = np.zeros(n)
        entry = np.zeros(n)
        steps_in = np.zeros(n)

        # flat throughout, so the position block stays at its reference values
        vec_obs0 = vec.observations(0, pos, entry, steps_in)
        for ep in range(n):
            env_obs, _ = env.reset(options={"episode": ep})
            np.testing.assert_allclose(vec_obs0[ep], env_obs, atol=1e-6)

    def test_unnormalised_observations_also_match(self, corpus) -> None:
        vec = VectorizedRollout(corpus, normalizer=None)
        env = BinaryMarketEnv(corpus, random_episode_order=False)
        n = len(corpus)
        z = np.zeros(n)
        vec_obs0 = vec.observations(0, z, z, z)
        for ep in range(n):
            env_obs, _ = env.reset(options={"episode": ep})
            np.testing.assert_allclose(vec_obs0[ep], env_obs, atol=1e-6)


class TestCostModes:
    def test_frictionless_parity(self, corpus) -> None:
        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm, costs_enabled=False)
        got = vec.run(lambda obs: np.full(len(obs), LONG))
        want = reference_run(
            corpus, lambda obs, t: np.full(len(obs), LONG), norm, costs_enabled=False
        )
        np.testing.assert_allclose(got["returns"], want["returns"], atol=1e-9)
        assert got["fees"].sum() == 0.0

    def test_costs_make_churn_worse_in_both_runners(self, corpus) -> None:
        norm = fit_normalizer(corpus)
        counter = {"t": 0}

        def churn(obs):
            a = LONG if counter["t"] % 2 == 0 else SHORT
            counter["t"] += 1
            return np.full(len(obs), a)

        with_costs = VectorizedRollout(corpus, normalizer=norm).run(churn)
        counter["t"] = 0
        without = VectorizedRollout(
            corpus, normalizer=norm, costs_enabled=False
        ).run(churn)
        assert with_costs["returns"].mean() < without["returns"].mean()


class TestNullCorpusParity:
    def test_parity_holds_on_the_null_corpus_too(self) -> None:
        """a different data distribution, in case the learnable one is special."""
        batch = make_null_corpus(n_episodes=30, n_steps=6, seed=3)
        norm = fit_normalizer(batch)
        vec = VectorizedRollout(batch, normalizer=norm)
        got = vec.run(lambda obs: np.full(len(obs), LONG))
        want = reference_run(batch, lambda obs, t: np.full(len(obs), LONG), norm)
        np.testing.assert_allclose(got["returns"], want["returns"], atol=1e-9)


class TestBackground:
    def test_background_shape_and_finiteness(self, corpus) -> None:
        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)
        bg = build_background(vec, n_samples=50, seed=0)
        assert bg.shape[0] == 50
        assert np.all(np.isfinite(bg))

    def test_background_position_block_is_flat(self, corpus) -> None:
        """the reference distribution should represent holding no inventory."""
        from nano_rl.env.features import MARKET_FEATURES

        norm = fit_normalizer(corpus)
        vec = VectorizedRollout(corpus, normalizer=norm)
        bg = build_background(vec, n_samples=40, seed=0)
        pos_block = bg[:, len(MARKET_FEATURES):]
        np.testing.assert_allclose(pos_block, 0.0, atol=1e-9)
