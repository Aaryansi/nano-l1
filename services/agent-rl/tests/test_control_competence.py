"""Failed learning cannot supply sensitivity evidence; actor exports preserve policies."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from nano_rl.envs.gym_null import (make_env, GymPPOConfig, train_gym_ppo,
                                 evaluate_gym, attribution_span_fast)
from nano_rl.agents.networks import ActorCritic
from nano_rl.envs.control_training import export_actor

spec = importlib.util.spec_from_file_location("generalize_gym", Path(__file__).resolve().parents[1] / "scripts/generalize_gym.py")
generalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generalize)


def test_competence_requires_return_and_observation_dependence():
    assert not generalize.competence_gate("CartPole-v1", 10., 100.)["eligible_for_sensitivity_claim"]
    assert not generalize.competence_gate("CartPole-v1", 500., 0.)["eligible_for_sensitivity_claim"]
    assert generalize.competence_gate("CartPole-v1", 500., 480.)["eligible_for_sensitivity_claim"]


@pytest.mark.parametrize("name", ["FrozenLake-v1", "CliffWalking-v0"])
def test_grid_coordinates_are_bounded_and_episodes_terminate(name):
    env = make_env(name)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert obs.shape == (2,)
        for _ in range(201):
            obs, _, terminated, truncated, _ = env.step(0)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        else:
            pytest.fail("grid evaluation did not reach an episode boundary")
    finally:
        env.close()


def test_export_preserves_action_probabilities_and_rng():
    env = make_env("CartPole-v1")
    try:
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1,
                    seed=3, device="cpu")
        model.learn(128)
        rng_before = torch.get_rng_state().clone()
        actor = export_actor(model.policy)
        assert torch.equal(rng_before, torch.get_rng_state())
        obs = torch.as_tensor(np.random.default_rng(6).normal(size=(32, 4)), dtype=torch.float32)
        with torch.no_grad():
            expected = model.policy.get_distribution(obs).distribution.probs
            actual = actor.action_probs(obs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    finally:
        env.close()


def test_independent_backend_snapshots_do_not_change_training():
    outputs = []
    for fractions in ((), (.5, 1.)):
        env = make_env("CartPole-v1")
        try:
            final, checkpoints = train_gym_ppo(env,
                GymPPOConfig(seed=7, backend="sb3", steps_per_batch=128,
                             minibatch_size=64, update_epochs=2),
                total_steps=512, checkpoint_fractions=fractions)
            outputs.append(final)
            if fractions:
                for a, b in zip(final.parameters(), checkpoints[-1][1].parameters()):
                    torch.testing.assert_close(a, b)
        finally:
            env.close()
    for a, b in zip(outputs[0].parameters(), outputs[1].parameters()):
        torch.testing.assert_close(a, b)


def test_intermediate_snapshot_includes_its_budget_optimizer_update():
    cfg = GymPPOConfig(seed=9, backend="sb3", steps_per_batch=128,
                       minibatch_size=64, update_epochs=2)
    short_env = make_env("CartPole-v1")
    long_env = make_env("CartPole-v1")
    try:
        short, _ = train_gym_ppo(short_env, cfg, total_steps=128)
        _, checkpoints = train_gym_ppo(long_env, cfg, total_steps=256,
                                       checkpoint_fractions=(.5, 1.))
        assert checkpoints[0][0] == .5
        for expected, actual in zip(short.parameters(), checkpoints[0][1].parameters()):
            torch.testing.assert_close(actual, expected)
    finally:
        short_env.close()
        long_env.close()


def test_control_endpoint_records_reconstruct_scores_with_common_episode_seeds():
    torch.manual_seed(15)
    actor = ActorCritic(4, 2, 64)
    background = np.random.default_rng(7).normal(size=(32, 4)).astype(np.float32)
    raw = attribution_span_fast(actor, "CartPole-v1", background, 4,
                               n_episodes=3, seed=11, return_samples=True)
    unmasked = evaluate_gym(actor, "CartPole-v1", n_episodes=3, seed=11, return_samples=True)
    np.testing.assert_array_equal(raw["full"], unmasked)
    assert raw["span"] == np.mean(raw["full"]) - np.mean(raw["masked"])
    assert raw["episode_seeds"] == [11, 12, 13]
    assert attribution_span_fast(actor, "CartPole-v1", background, 4,
                                 n_episodes=3, seed=11) == raw["span"]
