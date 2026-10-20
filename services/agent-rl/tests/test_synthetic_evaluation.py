"""Synthetic evaluation must be a separate reproducible population draw."""
import numpy as np
import pytest

from nano_rl.env.synthetic import (synthetic_train_eval, make_null_corpus,
                                   make_learnable_corpus)
from nano_rl.env.features import fit_normalizer
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.explain.rollout import VectorizedRollout, greedy_policy


@pytest.mark.parametrize("factory", [make_null_corpus, make_learnable_corpus])
def test_corpus_draws_are_independent_and_repeatable(factory):
    pair = synthetic_train_eval(factory, n_episodes=128, seed=42)
    again = synthetic_train_eval(factory, n_episodes=128, seed=42)
    assert pair.train_seed != pair.evaluation_seed
    assert not np.shares_memory(pair.train.spot, pair.evaluation.spot)
    assert not np.array_equal(pair.train.spot, pair.evaluation.spot)
    assert not np.array_equal(pair.train.settlement, pair.evaluation.settlement)
    np.testing.assert_array_equal(pair.evaluation.spot, again.evaluation.spot)
    assert pair.metadata()["evaluation_seed"] == 1000042


def test_fresh_agent_is_evaluated_on_independent_episodes():
    pair = synthetic_train_eval(make_learnable_corpus, n_episodes=32, n_steps=6, seed=7)
    normalizer = fit_normalizer(pair.train)
    env = BinaryMarketEnv(pair.train, normalizer=normalizer)
    env.reset(seed=7)
    agent = PPOAgent(18, 3, PPOConfig(seed=7, episodes_per_batch=8))
    agent.train(env, n_updates=2, verbose=False)
    evaluate = BinaryMarketEnv(pair.evaluation, normalizer=normalizer)
    scalar = agent.evaluate(evaluate, n_episodes=len(pair.evaluation))
    fast = VectorizedRollout(pair.evaluation, normalizer=normalizer).run(greedy_policy(agent))
    for key in fast:
        np.testing.assert_allclose(fast[key], scalar[key], atol=1e-8)
