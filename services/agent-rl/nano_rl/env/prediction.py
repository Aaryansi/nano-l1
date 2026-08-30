"""the same real episodes, scored on calling the outcome rather than trading it.

every case where our null test FIRES is otherwise synthetic, which invites the
obvious objection: perhaps the test only detects planted signal, and would
decline on any real corpus. this module supplies the positive control that
answers it, using nothing but real settled contracts.

the construction is deliberately minimal. the observations, the normalizer, the
episodes and the walk-forward split are all identical to the trading task. only
the objective changes:

    trading      reward is change in mark-to-market equity, net of the
                 exchange's fees. beating the market requires an edge over its
                 own price.

    prediction   reward is +1 for each step the agent's call agrees with the
                 eventual settlement, -1 for each step it disagrees, 0 for
                 abstaining. agreeing with the market's price is enough.

the difference matters because the market's implied probability is well
calibrated (weighted mean absolute error 0.0172). a calibrated price is highly
informative ABOUT THE OUTCOME while offering nothing to trade against, since
the fee schedule takes more than the edge. so the same feature stream should
carry information on the prediction task and none on the trading task, and the
test should say so.

this turns a single negative result into a matched pair on identical data. if
the test fired on both we would have learned nothing; if it declined on both,
the test would look inert. the informative outcome is the one that separates
them, and it is the one we get.

reward here is allowed to depend on the settlement, which is not lookahead: an
objective may reference the outcome, an OBSERVATION may not. nothing in the
18-feature state vector sees the future, and the no-lookahead suite still
governs the features this env serves.
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
    """BinaryMarketEnv with the trading objective replaced by a scored call.

    subclassed rather than reimplemented so the observation construction, the
    normalizer handling and the episode ordering are literally the same code.
    if they were merely similar, a difference in results could be a difference
    in the environment rather than in the objective, which is the one thing
    this control exists to rule out.

    the agent's current call occupies the position block of the observation,
    which is the natural analogue of inventory and keeps the state 18-dimensional
    so both tasks are explained by identical machinery.
    """

    def step(self, action: int):  # type: ignore[override]
        call = float(ACTION_TO_TARGET[int(action)])
        mark = float(
            0.5
            * (
                self.batch.bid[self._ep, self._t]
                + self.batch.ask[self._ep, self._t]
            )
        )

        # the call is carried in the position slot; there is no execution, no
        # fill and no fee, so nothing touches the cost model.
        if call != 0.0 and self._pos.position != call * self.max_position:
            self._pos.steps_in_position = 0
        self._pos.position = call * self.max_position
        self._pos.avg_entry_price = mark if call != 0.0 else 0.0
        self._pos.steps_in_position += 1 if call != 0.0 else 0
        if call == 0.0:
            self._pos.steps_in_position = 0

        reward = call * float(outcome_sign(self.batch.settlement[self._ep]))

        self._t += 1
        terminated = self._t >= self.n_steps

        info = {
            "call": call,
            "position": self._pos.position,
            "correct": float(reward > 0.0),
            "settlement": float(self.batch.settlement[self._ep]),
        }
        return self._obs(), float(reward), terminated, False, info


class PredictionRollout(VectorizedRollout):
    """batched replay of the prediction task, for attribution.

    mirrors VectorizedRollout.run's interface exactly, returning a "returns"
    key, so every masking and attribution utility in nano_rl.explain works on
    this task with no changes.
    """

    def run(self, policy) -> dict[str, np.ndarray]:  # type: ignore[override]
        n = self.n_episodes
        sign = outcome_sign(self.batch.settlement)

        total = np.zeros(n)
        correct = np.zeros(n)
        calls = np.zeros(n)
        avg_entry = np.zeros(n)
        steps_in = np.zeros(n)

        for t in range(self.n_steps):
            obs = self.observations(
                t, calls * self.max_position, avg_entry, steps_in
            )
            new_calls = _CALLS[np.asarray(policy(obs), dtype=int)]

            mark = 0.5 * (self.batch.bid[:, t] + self.batch.ask[:, t])
            # mirrors PredictionEnv.step exactly: abstaining resets the counter
            # to zero, but switching to a NEW call starts it at one, because the
            # env increments after assigning the position. an earlier version
            # reset to zero on a change, which left this feature one step behind
            # the env for the whole episode. constant-policy parity tests cannot
            # see that, since a constant policy ignores its observation.
            steps_in = np.where(
                new_calls == 0.0, 0.0,
                np.where(new_calls != calls, 1.0, steps_in + 1.0),
            )
            avg_entry = np.where(new_calls != 0.0, mark, 0.0)
            calls = new_calls

            step_reward = calls * sign
            total += step_reward
            correct += (step_reward > 0.0).astype(float)

        return {
            "returns": total,
            "correct": correct,
            "trades": np.zeros(n),
            "fees": np.zeros(n),
            "final_position": calls * self.max_position,
        }


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
