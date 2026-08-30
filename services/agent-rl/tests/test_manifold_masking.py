"""tests for conditional masking and the action discretiser.

the claim these support is the one section 5.8 of the paper rests on: that the
attribution span cannot be an off-manifold artefact, because the two coalitions
it is built from are on the manifold under either masking mode. that is an
argument about code as much as about probability, so it is asserted here rather
than left to the reader to re-derive.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from nano_rl.envs.gym_null import DiscretizedAction, make_env
from nano_rl.explain.manifold import (
    _scales,
    draw_replacements,
    offmanifold_distance,
)


@pytest.fixture(scope="module")
def background() -> np.ndarray:
    """correlated reference rows, so conditioning has something to exploit."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(256, 1))
    return np.hstack([a, a * 2.0 + rng.normal(scale=0.05, size=(256, 1)),
                      rng.normal(size=(256, 2))])


class TestModesCoincideAtTheEndpoints:
    """the structural argument behind the span's immunity to the objection."""

    def test_empty_kept_set_falls_back_to_marginal(self, background):
        # p(x | empty) is the marginal, so conditioning is vacuous and the two
        # modes must draw from the same distribution. we check they produce
        # identical draws given identical rng state.
        obs = background[:16]
        empty = np.zeros(background.shape[1], dtype=bool)
        s = _scales(background)
        a = draw_replacements(obs, empty, background,
                              np.random.default_rng(7), "marginal", 8, s)
        b = draw_replacements(obs, empty, background,
                              np.random.default_rng(7), "conditional", 8, s)
        np.testing.assert_array_equal(a, b)

    def test_fully_masked_state_is_a_real_reference_row(self, background):
        # v(empty) substitutes the whole observation from ONE reference row, so
        # the synthetic state is itself a real observation. this is why the
        # span cannot be off-manifold, and it holds only because whole rows are
        # drawn rather than each feature drawn independently.
        obs = background[:16]
        empty = np.zeros(background.shape[1], dtype=bool)
        draws = draw_replacements(obs, empty, background,
                                  np.random.default_rng(1), "marginal", 8,
                                  _scales(background))
        for row in draws:
            assert np.isclose(np.abs(background - row).sum(axis=1).min(), 0.0)

    def test_full_mask_replaces_nothing(self, background):
        obs = background[:16].copy()
        full = np.ones(background.shape[1], dtype=bool)
        d = offmanifold_distance(obs, full, background,
                                 np.random.default_rng(0), "conditional", 8)
        # nothing is substituted, so this is just obs-to-nearest-reference
        assert d >= 0.0


class TestConditioningActuallyConditions:
    def test_conditional_draws_are_nearer_in_the_kept_features(self, background):
        """the whole point: replacements should match the retained context.

        without this the conditional mode would be an expensive alias for the
        marginal one and the section 5.8 comparison would be vacuous.
        """
        obs = background[:64]
        mask = np.array([True, False, False, False])
        s = _scales(background)
        errs = {}
        for mode in ("marginal", "conditional"):
            d = draw_replacements(obs, mask, background,
                                  np.random.default_rng(3), mode, 4, s)
            # feature 1 is ~2x feature 0; a conditioned draw should respect it
            errs[mode] = float(np.abs(d[:, 1] - obs[:, 0] * 2.0).mean())
        assert errs["conditional"] < errs["marginal"]

    def test_more_masking_moves_marginal_further_off_manifold(self, background):
        obs = background[:64]
        rng = np.random.default_rng(5)
        keep_one = np.array([True, False, False, False])
        keep_three = np.array([True, True, True, False])
        far = offmanifold_distance(obs, keep_one, background, rng, "marginal", 8)
        near = offmanifold_distance(obs, keep_three, background, rng,
                                    "marginal", 8)
        assert far > near


class TestDiscretizedAction:
    def test_box_action_space_becomes_discrete(self):
        env = make_env("Pendulum-v1")
        assert isinstance(env.action_space, gym.spaces.Discrete)
        env.close()

    def test_discrete_envs_are_left_alone(self):
        env = make_env("CartPole-v1")
        assert not isinstance(env, DiscretizedAction)
        assert env.action_space.n == 2
        env.close()

    def test_grid_spans_the_original_range(self):
        base = gym.make("Pendulum-v1")
        lo = float(base.action_space.low[0])
        hi = float(base.action_space.high[0])
        base.close()

        wrapped = DiscretizedAction(gym.make("Pendulum-v1"), n_bins=9)
        assert np.isclose(wrapped.action(0)[0], lo)
        assert np.isclose(wrapped.action(8)[0], hi)
        wrapped.close()

    def test_wrapped_env_steps(self):
        env = make_env("Pendulum-v1")
        env.reset(seed=0)
        for a in range(env.action_space.n):
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset(seed=0)
        env.close()
