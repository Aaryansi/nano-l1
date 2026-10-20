"""Trading-state observations scored against settlement rather than P&L.

Prediction and trading use exactly the same action effects and observation
construction. The prediction score replaces only reward and excludes shadow
execution costs. Settlement is allowed in reward and never enters observations.
"""

from __future__ import annotations

import numpy as np

from nano_rl.env.binary_market import ACTION_TO_TARGET, BinaryMarketEnv
from nano_rl.explain.rollout import VectorizedRollout

# {-1, 0, +1} indexed by action, as an array for the vectorised path
_CALLS = np.asarray(ACTION_TO_TARGET, dtype=float)


def outcome_sign(settlement: np.ndarray) -> np.ndarray:
    """map settlement in {0, 1} to {-1, +1}."""
    return 2.0 * np.asarray(settlement, dtype=float) - 1.0


class PredictionEnv(BinaryMarketEnv):
    """Identical trading state transitions, scored for calling settlement.

    The shadow inventory executes exactly as in the trading task, including
    entry price and holding time. Only the reward used for learning differs.
    Shadow execution costs are recorded in info but do not enter this score.
    """

    def step(self, action: int):
        observation, _, terminated, truncated, info = super().step(action)
        call = float(ACTION_TO_TARGET[int(action)])
        reward = call * float(outcome_sign(self.batch.settlement[self._ep]))
        info.update(call=call, correct=float(reward > 0),
                    settlement=float(self.batch.settlement[self._ep]))
        return observation, reward, terminated, truncated, info


class PredictionRollout(VectorizedRollout):
    """Use shared trading transitions and replace only the objective score."""

    def score_step(self, actions, t):
        return _CALLS[actions] * outcome_sign(self.batch.settlement)


class BlindEnv:
    """replace every observation with a draw from a fixed reference distribution.

    the environment-level null of Section 3, applied to a real corpus rather
    than to a synthetic one. dynamics, episodes and reward are untouched; the
    agent trains normally and simply cannot condition on anything.

    written as an explicit delegating wrapper rather than a
    gymnasium.ObservationWrapper because gymnasium 1.0 stopped forwarding
    arbitrary attributes through wrappers, and the agent code reaches for
    `env.batch`.
    """

    def __init__(self, env: BinaryMarketEnv, mean: np.ndarray, std: np.ndarray,
                 seed: int = 0) -> None:
        self.env = env
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    # attributes the agent and the rollout utilities reach for
    @property
    def batch(self):
        return self.env.batch

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def _blind(self) -> np.ndarray:
        return self._rng.normal(self._mean, self._std).astype(np.float32)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._blind(), info

    def step(self, action):
        _, r, term, trunc, info = self.env.step(action)
        return self._blind(), r, term, trunc, info


def observation_moments(env: BinaryMarketEnv, n_steps: int = 4000,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """mean and sd of the observation under a random policy.

    matching these matters: a network fed out-of-range inputs fails for reasons
    of scale rather than of information, which would be a different experiment.
    """
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    for _ in range(n_steps):
        rows.append(np.asarray(obs, dtype=np.float64))
        obs, _, term, trunc, _ = env.step(int(rng.integers(0, 3)))
        if term or trunc:
            obs, _ = env.reset()
    arr = np.asarray(rows)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-8
