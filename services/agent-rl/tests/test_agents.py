"""agent tests.

the expensive correctness checks live in scripts/sanity_tabular.py and
scripts/sanity_ppo.py, which train real agents against corpora with known
answers. those take minutes, so they are not unit tests.

what is tested here is the machinery those checks depend on: gae arithmetic,
the ppo update's shape and stability, determinism under a fixed seed, and the
normalizer guard that caught a silent learning failure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from nano_rl.agents.networks import ActorCritic
from nano_rl.agents.ppo import PPOAgent, PPOConfig, RolloutBuffer
from nano_rl.agents.tabular_q import TabularQAgent, TabularQConfig
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.features import N_FEATURES, fit_normalizer
from nano_rl.env.synthetic import (
    SIGNAL_IDX,
    discretize,
    flat_policy_return,
    make_learnable_corpus,
    make_null_corpus,
    signal_policy_return,
)


@pytest.fixture
def small_learnable():
    return make_learnable_corpus(n_episodes=60, n_steps=6, seed=0)


class TestGAE:
    """gae arithmetic, checked against hand-computed values."""

    def test_single_step_episode(self) -> None:
        buf = RolloutBuffer()
        buf.start_episode()
        buf.add(np.zeros(3), 0, 0.0, value=2.0, reward=5.0)
        adv, ret = buf.compute_gae(gamma=1.0, lam=1.0)
        # terminal, so next_value = 0: delta = 5 + 0 - 2 = 3
        assert adv[0] == pytest.approx(3.0)
        assert ret[0] == pytest.approx(5.0)

    def test_terminal_value_is_zero_not_bootstrapped(self) -> None:
        """episodes end at a true terminal; bootstrapping there would be wrong."""
        buf = RolloutBuffer()
        buf.start_episode()
        buf.add(np.zeros(3), 0, 0.0, value=10.0, reward=0.0)
        adv, _ = buf.compute_gae(gamma=1.0, lam=1.0)
        # delta = 0 + 0 - 10 = -10, not 0 + 10 - 10 = 0
        assert adv[0] == pytest.approx(-10.0)

    def test_two_step_lambda_one_is_monte_carlo(self) -> None:
        buf = RolloutBuffer()
        buf.start_episode()
        buf.add(np.zeros(3), 0, 0.0, value=0.0, reward=1.0)
        buf.add(np.zeros(3), 0, 0.0, value=0.0, reward=2.0)
        adv, ret = buf.compute_gae(gamma=1.0, lam=1.0)
        # with zero baselines and lambda=1, advantage == return-to-go
        assert adv[0] == pytest.approx(3.0)
        assert adv[1] == pytest.approx(2.0)

    def test_episodes_do_not_bleed_into_each_other(self) -> None:
        """a second episode's rewards must not propagate into the first."""
        buf = RolloutBuffer()
        buf.start_episode()
        buf.add(np.zeros(3), 0, 0.0, value=0.0, reward=1.0)
        buf.start_episode()
        buf.add(np.zeros(3), 0, 0.0, value=0.0, reward=100.0)
        adv, _ = buf.compute_gae(gamma=1.0, lam=1.0)
        assert adv[0] == pytest.approx(1.0)
        assert adv[1] == pytest.approx(100.0)

    def test_returns_equal_advantages_plus_values(self) -> None:
        rng = np.random.default_rng(0)
        buf = RolloutBuffer()
        buf.start_episode()
        for _ in range(8):
            buf.add(np.zeros(3), 0, 0.0, float(rng.normal()), float(rng.normal()))
        adv, ret = buf.compute_gae(gamma=1.0, lam=0.95)
        np.testing.assert_allclose(ret, adv + np.asarray(buf.values), atol=1e-9)

    def test_episode_returns_sum_rewards(self) -> None:
        buf = RolloutBuffer()
        buf.start_episode()
        for r in (1.0, 2.0, 3.0):
            buf.add(np.zeros(3), 0, 0.0, 0.0, r)
        buf.start_episode()
        buf.add(np.zeros(3), 0, 0.0, 0.0, -5.0)
        np.testing.assert_allclose(buf.episode_returns(), [6.0, -5.0])


class TestNetwork:
    def test_output_shapes(self) -> None:
        net = ActorCritic(N_FEATURES, 3)
        x = torch.zeros(7, N_FEATURES)
        logits, value = net(x)
        assert logits.shape == (7, 3)
        assert value.shape == (7,)

    def test_initial_policy_is_near_uniform(self) -> None:
        """small output gain keeps early training from committing blind."""
        net = ActorCritic(N_FEATURES, 3)
        probs = net.action_probs(torch.zeros(1, N_FEATURES))[0]
        assert probs.max().item() < 0.40  # uniform is 0.333

    def test_deterministic_act_is_argmax(self) -> None:
        net = ActorCritic(N_FEATURES, 3)
        x = torch.randn(5, N_FEATURES)
        a, _, _ = net.act(x, deterministic=True)
        expected = net(x)[0].argmax(dim=-1)
        torch.testing.assert_close(a, expected)

    def test_evaluate_actions_matches_forward(self) -> None:
        net = ActorCritic(N_FEATURES, 3)
        x = torch.randn(4, N_FEATURES)
        acts = torch.tensor([0, 1, 2, 0])
        logp, ent, val = net.evaluate_actions(x, acts)
        assert logp.shape == (4,)
        assert ent.shape == (4,)
        assert val.shape == (4,)
        assert torch.all(ent >= 0)


class TestPPOUpdate:
    def test_update_runs_and_reports_finite_stats(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)
        env = BinaryMarketEnv(small_learnable, normalizer=norm)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=8))
        buf = agent.collect(env, 8)
        stats = agent.update(buf)
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            assert np.isfinite(stats[k]), f"{k} not finite"

    def test_action_frequencies_sum_to_one(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)
        env = BinaryMarketEnv(small_learnable, normalizer=norm)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=8))
        stats = agent.update(agent.collect(env, 8))
        assert sum(stats["action_freq"]) == pytest.approx(1.0)

    def test_training_is_deterministic_under_seed(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)

        def run() -> list[float]:
            env = BinaryMarketEnv(small_learnable, normalizer=norm)
            env.reset(seed=0)
            agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=8))
            return agent.train(env, n_updates=3, verbose=False).mean_return

        np.testing.assert_allclose(run(), run())

    def test_different_seeds_diverge(self, small_learnable) -> None:
        """guards against a seed that is accidentally ignored."""
        norm = fit_normalizer(small_learnable)

        def run(seed: int) -> list[float]:
            env = BinaryMarketEnv(small_learnable, normalizer=norm)
            env.reset(seed=seed)
            agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed, episodes_per_batch=8))
            return agent.train(env, n_updates=3, verbose=False).mean_return

        assert not np.allclose(run(0), run(1))

    def test_lr_anneals_to_near_zero(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)
        env = BinaryMarketEnv(small_learnable, normalizer=norm)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=4))
        log = agent.train(env, n_updates=5, verbose=False)
        assert log.lr[0] > log.lr[-1]

    def test_log_lengths_match_update_count(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)
        env = BinaryMarketEnv(small_learnable, normalizer=norm)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=4))
        log = agent.train(env, n_updates=4, verbose=False)
        for series in (log.mean_return, log.entropy, log.approx_kl, log.explained_var):
            assert len(series) == 4

    def test_save_and_load_roundtrip(self, small_learnable, tmp_path) -> None:
        norm = fit_normalizer(small_learnable)
        env = BinaryMarketEnv(small_learnable, normalizer=norm, random_episode_order=False)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, episodes_per_batch=4))
        agent.train(env, n_updates=2, verbose=False)

        p = tmp_path / "agent.pt"
        agent.save(str(p))

        fresh = PPOAgent(N_FEATURES, 3, PPOConfig(seed=99))
        fresh.load(str(p))

        x = torch.randn(3, N_FEATURES)
        torch.testing.assert_close(agent.net.action_probs(x), fresh.net.action_probs(x))


class TestNormalizerGuard:
    """regression test for the bug that pinned ppo at 0.00 return.

    volume_rate reaches log1p(10000) = 9.21, which saturated 31% of the first
    tanh layer. the failure is dangerous because a saturated network that never
    learns is indistinguishable from an agent correctly finding no signal.
    """

    def test_warns_when_unnormalized_features_are_large(self, small_learnable) -> None:
        with pytest.warns(RuntimeWarning, match="without a normalizer"):
            BinaryMarketEnv(small_learnable)

    def test_no_warning_when_normalizer_supplied(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BinaryMarketEnv(small_learnable, normalizer=norm)

    def test_normalisation_actually_shrinks_the_range(self, small_learnable) -> None:
        norm = fit_normalizer(small_learnable)
        env = BinaryMarketEnv(small_learnable, normalizer=norm, random_episode_order=False)
        obs, _ = env.reset(seed=0, options={"episode": 0})
        assert np.abs(obs).max() < 5.0


class TestTabularQ:
    def test_epsilon_decays_monotonically(self) -> None:
        agent = TabularQAgent(TabularQConfig(eps_decay_episodes=100))
        first = agent.epsilon()
        agent._episode = 100
        assert agent.epsilon() < first
        assert agent.epsilon() == pytest.approx(agent.cfg.eps_end)

    def test_epsilon_floors_at_end_value(self) -> None:
        agent = TabularQAgent(TabularQConfig(eps_decay_episodes=10))
        agent._episode = 10_000
        assert agent.epsilon() == pytest.approx(agent.cfg.eps_end)

    def test_greedy_ignores_epsilon(self) -> None:
        agent = TabularQAgent(TabularQConfig(eps_start=1.0, eps_end=1.0))
        agent.q[0] = [0.0, 0.0, 5.0]
        assert all(agent.act(0, greedy=True) == 2 for _ in range(20))

    def test_update_moves_toward_target(self) -> None:
        agent = TabularQAgent(TabularQConfig(lr=0.5))
        agent.update(s=0, a=1, r=10.0, s2=0, done=True)
        assert agent.q[0, 1] == pytest.approx(5.0)

    def test_terminal_update_ignores_next_state(self) -> None:
        agent = TabularQAgent(TabularQConfig(lr=1.0))
        agent.q[1] = [100.0, 100.0, 100.0]
        agent.update(s=0, a=0, r=1.0, s2=1, done=True)
        assert agent.q[0, 0] == pytest.approx(1.0)


class TestDiscretize:
    def test_monotonic_in_the_signal(self) -> None:
        obs = np.zeros(N_FEATURES)
        prev = -1
        for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
            obs[SIGNAL_IDX] = v
            b = discretize(obs, SIGNAL_IDX)
            assert b >= prev
            prev = b

    def test_stays_in_range(self) -> None:
        obs = np.zeros(N_FEATURES)
        for v in (-1e6, 1e6):
            obs[SIGNAL_IDX] = v
            assert 0 <= discretize(obs, SIGNAL_IDX, n_bins=9) <= 9


class TestBenchmarks:
    def test_signal_policy_beats_flat_on_learnable(self) -> None:
        b = make_learnable_corpus(n_episodes=500, seed=0)
        assert signal_policy_return(b) > 10.0

    def test_signal_policy_loses_on_null(self) -> None:
        """following noise pays frictions, so it must underperform flat."""
        b = make_null_corpus(n_episodes=500, seed=0)
        assert signal_policy_return(b) < flat_policy_return()

    def test_costs_reduce_the_benchmark(self) -> None:
        b = make_learnable_corpus(n_episodes=500, seed=0)
        assert signal_policy_return(b, with_costs=True) < signal_policy_return(
            b, with_costs=False
        )

    def test_null_corpus_signal_is_uninformative(self) -> None:
        b = make_null_corpus(n_episodes=2000, seed=0)
        sig = b.spot[:, 0, SIGNAL_IDX]
        corr = np.corrcoef(sig, b.settlement)[0, 1]
        assert abs(corr) < 0.1

    def test_learnable_corpus_signal_is_informative(self) -> None:
        b = make_learnable_corpus(n_episodes=2000, seed=0)
        sig = b.spot[:, 0, SIGNAL_IDX]
        corr = np.corrcoef(sig, b.settlement)[0, 1]
        assert corr > 0.7
