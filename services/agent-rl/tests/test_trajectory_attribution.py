"""tests for the three attribution targets.

the heavy validation lives in scripts/explain.py, which trains an agent on a
corpus with a planted signal and checks the attribution finds it. these are the
unit-level guarantees that make that script's output interpretable: correct
shapes, the efficiency identity, and the comparison utility behaving sensibly
in the cases where the answer is obvious.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.features import FEATURE_NAMES, N_FEATURES, fit_normalizer
from nano_rl.env.synthetic import make_learnable_corpus
from nano_rl.explain.rollout import VectorizedRollout, build_background
from nano_rl.explain.shapley import Attribution
from nano_rl.explain.trajectory import (
    OutcomeAttributionConfig,
    compare_naive_and_trajectory,
    explain_behaviour,
    explain_outcomes,
    explain_value,
)


@pytest.fixture(scope="module")
def setup():
    batch = make_learnable_corpus(n_episodes=60, n_steps=6, seed=0)
    norm = fit_normalizer(batch)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=8))
    env = BinaryMarketEnv(batch, normalizer=norm, max_position=100.0)
    env.reset(seed=0)
    agent.train(env, n_updates=3, verbose=False)

    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    background = build_background(roll, n_samples=64, seed=0)

    eval_env = BinaryMarketEnv(
        batch, normalizer=norm, max_position=100.0, random_episode_order=False
    )
    obs, _ = eval_env.reset(options={"episode": 0})
    return agent, batch, norm, background, obs


class TestBehaviourTarget:
    def test_shape_and_names(self, setup) -> None:
        agent, _, _, bg, obs = setup
        att, action = explain_behaviour(agent, obs, bg, n_permutations=20, seed=0)
        assert att.values.shape == (N_FEATURES,)
        assert att.feature_names == FEATURE_NAMES
        assert action in (0, 1, 2)

    def test_explains_the_agents_own_choice_by_default(self, setup) -> None:
        agent, _, _, bg, obs = setup
        import torch

        with torch.no_grad():
            logits, _ = agent.net(torch.as_tensor(obs, dtype=torch.float32)[None, :])
        _, action = explain_behaviour(agent, obs, bg, n_permutations=10, seed=0)
        assert action == int(logits.argmax().item())

    def test_can_explain_a_counterfactual_action(self, setup) -> None:
        """asking why it did NOT do something is a legitimate question."""
        agent, _, _, bg, obs = setup
        att, action = explain_behaviour(agent, obs, bg, action=0, n_permutations=10)
        assert action == 0
        assert np.all(np.isfinite(att.values))

    def test_probability_bounds_the_base_and_full_values(self, setup) -> None:
        agent, _, _, bg, obs = setup
        att, _ = explain_behaviour(agent, obs, bg, n_permutations=20, seed=0)
        for v in (att.base_value, att.full_value):
            assert 0.0 <= v <= 1.0


class TestValueTarget:
    def test_shape_and_finiteness(self, setup) -> None:
        agent, _, _, bg, obs = setup
        att = explain_value(agent, obs, bg, n_permutations=20, seed=0)
        assert att.values.shape == (N_FEATURES,)
        assert np.all(np.isfinite(att.values))

    def test_efficiency_holds_approximately(self, setup) -> None:
        agent, _, _, bg, obs = setup
        att = explain_value(agent, obs, bg, n_permutations=120, seed=0)
        assert att.relative_efficiency_gap < 0.05


class TestOutcomeTarget:
    def test_shape_and_exact_efficiency(self, setup) -> None:
        agent, batch, norm, bg, _ = setup
        att = explain_outcomes(
            agent, batch, bg, normalizer=norm,
            cfg=OutcomeAttributionConfig(n_coalitions=40, n_episodes=40, seed=0),
        )
        assert att.values.shape == (N_FEATURES,)
        # kernel shap imposes efficiency by substitution
        assert att.efficiency_gap < 1e-8

    def test_base_value_is_the_blind_agents_return(self, setup) -> None:
        """v(empty) must equal what the agent earns seeing nothing real."""
        agent, batch, norm, bg, _ = setup
        att = explain_outcomes(
            agent, batch, bg, normalizer=norm,
            cfg=OutcomeAttributionConfig(n_coalitions=20, n_episodes=40, seed=0),
        )
        assert np.isfinite(att.base_value)
        assert np.isfinite(att.full_value)

    def test_costs_can_be_disabled(self, setup) -> None:
        agent, batch, norm, bg, _ = setup
        att = explain_outcomes(
            agent, batch, bg, normalizer=norm,
            cfg=OutcomeAttributionConfig(
                n_coalitions=20, n_episodes=40, seed=0, costs_enabled=False
            ),
        )
        assert np.all(np.isfinite(att.values))


class TestComparison:
    def _att(self, values: list[float]) -> Attribution:
        v = np.array(values, dtype=float)
        return Attribution(
            values=v,
            stderr=np.zeros_like(v),
            base_value=0.0,
            full_value=float(v.sum()),
            feature_names=tuple(f"f{i}" for i in range(len(v))),
        )

    def test_identical_attributions_correlate_perfectly(self) -> None:
        a = self._att([3.0, 1.0, -2.0, 0.5])
        cmp = compare_naive_and_trajectory(a, a, top_k=2)
        assert cmp["rank_correlation"] == pytest.approx(1.0)
        assert cmp["top_k_overlap"] == pytest.approx(1.0)

    def test_reversed_rankings_correlate_negatively(self) -> None:
        a = self._att([4.0, 3.0, 2.0, 1.0])
        b = self._att([1.0, 2.0, 3.0, 4.0])
        cmp = compare_naive_and_trajectory(a, b, top_k=2)
        assert cmp["rank_correlation"] < 0

    def test_reports_features_unique_to_each_view(self) -> None:
        a = self._att([5.0, 0.0, 0.0, 1.0])
        b = self._att([0.0, 5.0, 1.0, 0.0])
        cmp = compare_naive_and_trajectory(a, b, top_k=2)
        assert "f0" in cmp["only_in_naive"]
        assert "f1" in cmp["only_in_trajectory"]

    def test_ranks_by_absolute_value_not_sign(self) -> None:
        """a large negative contribution is still a large contribution."""
        a = self._att([-9.0, 1.0, 0.1])
        assert a.top(1)[0][0] == "f0"

    def test_degenerate_all_zero_gives_nan_not_a_crash(self) -> None:
        a = self._att([0.0, 0.0, 0.0])
        cmp = compare_naive_and_trajectory(a, a, top_k=2)
        assert np.isnan(cmp["rank_correlation"])
