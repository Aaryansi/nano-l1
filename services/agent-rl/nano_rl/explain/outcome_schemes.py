"""outcome-level attribution under three credit-assignment schemes.

a clarification first, because the paper was imprecise about it. the null
test's statistic is

    span = v(all features) - v(no features)

and that is **not a Shapley quantity**. it is the difference between two masked
rollouts. Shapley's efficiency axiom says the Shapley values happen to sum to
it, which is why the span can be read off an attribution, but the span itself
depends only on the masking scheme. so "is the null-test result Shapley
specific?" is partly answered by construction: no, because the statistic never
involved Shapley.

what genuinely could be method specific is the per-feature decomposition, and
whether a scheme with a DIFFERENT total reaches the same verdict. this module
supplies two such schemes. both are perturbation based and outcome level, like
Shapley, and neither satisfies efficiency, so each carries its own total:

    leave-one-out   phi_i = v(N) - v(N \\ {i})
                    what is lost by removing feature i from everything else.

    only-one-in     phi_i = v({i}) - v(empty)
                    what feature i is worth on its own.

Shapley is the weighted average of marginal contributions across all coalition
sizes; LOO and OOI are the two extremes of that average. if all three reach the
same verdict, the finding does not depend on how credit is distributed.

they are also cheaper: n+1 evaluations each against kernel Shapley's hundreds.
"""

from __future__ import annotations

import numpy as np
import torch

from nano_rl.agents.ppo import PPOAgent
from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.features import N_FEATURES, FeatureNormalizer
from nano_rl.explain.rollout import VectorizedRollout


def _masked_value_fn(
    agent: PPOAgent,
    batch: EpisodeBatch,
    background: np.ndarray,
    normalizer: FeatureNormalizer | None,
    max_position: float,
    n_episodes: int,
    seed: int,
):
    """v(S): mean episode return when the agent sees only the features in S.

    identical masking to nano_rl/explain/trajectory.py, so the three schemes
    and Shapley are all measured against the same reference. using a different
    masking here would confound scheme with masking, which is the mistake the
    integrated-gradients comparison already made once.
    """
    sub = batch.subset(np.arange(min(n_episodes, len(batch))))
    roll = VectorizedRollout(sub, normalizer=normalizer, max_position=max_position)
    rng = np.random.default_rng(seed)

    def v(mask: np.ndarray) -> float:
        def policy(obs: np.ndarray) -> np.ndarray:
            synthetic = obs.copy()
            if not mask.all():
                draws = background[rng.integers(0, len(background), size=len(obs))]
                synthetic[:, ~mask] = draws[:, ~mask]
            with torch.no_grad():
                logits, _ = agent.net(
                    torch.as_tensor(synthetic, dtype=torch.float32)
                )
                return logits.argmax(dim=-1).numpy()

        return float(roll.run(policy)["returns"].mean())

    return v


def leave_one_out(
    agent: PPOAgent,
    batch: EpisodeBatch,
    background: np.ndarray,
    normalizer: FeatureNormalizer | None = None,
    max_position: float = 100.0,
    n_episodes: int = 250,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """phi_i = v(N) - v(N minus i). returns (values, total).

    the total is the sum of the values, which for this scheme is NOT the span:
    with redundant features every individual removal costs little while
    removing all of them costs a great deal. that difference is the point of
    including it.
    """
    v = _masked_value_fn(
        agent, batch, background, normalizer, max_position, n_episodes, seed
    )
    full = np.ones(N_FEATURES, dtype=bool)
    v_full = v(full)

    values = np.empty(N_FEATURES)
    for i in range(N_FEATURES):
        m = full.copy()
        m[i] = False
        values[i] = v_full - v(m)
    return values, float(values.sum())


def only_one_in(
    agent: PPOAgent,
    batch: EpisodeBatch,
    background: np.ndarray,
    normalizer: FeatureNormalizer | None = None,
    max_position: float = 100.0,
    n_episodes: int = 250,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """phi_i = v({i}) - v(empty). returns (values, total)."""
    v = _masked_value_fn(
        agent, batch, background, normalizer, max_position, n_episodes, seed
    )
    v_empty = v(np.zeros(N_FEATURES, dtype=bool))

    values = np.empty(N_FEATURES)
    for i in range(N_FEATURES):
        m = np.zeros(N_FEATURES, dtype=bool)
        m[i] = True
        values[i] = v(m) - v_empty
    return values, float(values.sum())


def span_only(
    agent: PPOAgent,
    batch: EpisodeBatch,
    background: np.ndarray,
    normalizer: FeatureNormalizer | None = None,
    max_position: float = 100.0,
    n_episodes: int = 250,
    seed: int = 0,
) -> float:
    """v(all) - v(none), computed directly. scheme independent by construction."""
    v = _masked_value_fn(
        agent, batch, background, normalizer, max_position, n_episodes, seed
    )
    return v(np.ones(N_FEATURES, dtype=bool)) - v(np.zeros(N_FEATURES, dtype=bool))
