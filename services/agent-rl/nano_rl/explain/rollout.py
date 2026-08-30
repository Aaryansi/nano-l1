"""vectorised episode rollout, for attribution work that needs many replays.

trajectory-aware attribution evaluates a characteristic function whose value is
an expected episode return. each evaluation replays a batch of episodes, and
shapley needs hundreds of evaluations, so the per-step python overhead of the
gym env becomes the binding constraint.

this module runs N episodes in lockstep so the policy sees one (N, n_features)
batch per step instead of N separate forward passes. that turns 14*N network
calls into 14.

the accounting is duplicated from nano_rl/env/binary_market.py, which is a real
risk: two implementations of the same pnl logic can drift apart silently and
the fast one would quietly produce wrong explanations. so
tests/test_rollout_parity.py asserts the two agree to machine precision on
every policy it can construct. treat that test as load-bearing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nano_rl.env.binary_market import ACTION_TO_TARGET, EpisodeBatch
from nano_rl.env.costs import TAKER_FEE_COEF
from nano_rl.env.features import N_KALSHI, FeatureNormalizer

# a batched policy maps (n_episodes, n_features) observations to (n_episodes,)
# integer actions.
BatchPolicy = Callable[[np.ndarray], np.ndarray]


def _fee(n_contracts: np.ndarray, price: np.ndarray, coef: float = TAKER_FEE_COEF) -> np.ndarray:
    """vectorised kalshi fee, rounded up to the next cent PER ORDER.

    mirrors nano_rl.env.costs.fee_dollars. the round(_, 9) guards the same
    binary-representation edge case: without it, a value that is exactly a cent
    can be nudged upward and charge an extra cent.
    """
    raw = coef * np.abs(n_contracts) * price * (1.0 - price)
    cents = np.ceil(np.round(raw * 100.0, 9))
    return np.where(n_contracts == 0.0, 0.0, cents / 100.0)


class VectorizedRollout:
    """replay a whole EpisodeBatch under a batched policy."""

    def __init__(
        self,
        batch: EpisodeBatch,
        normalizer: FeatureNormalizer | None = None,
        max_position: float = 100.0,
        costs_enabled: bool = True,
    ) -> None:
        self.batch = batch
        self.normalizer = normalizer
        self.max_position = max_position
        self.costs_enabled = costs_enabled

        self.n_episodes = len(batch)
        self.n_steps = batch.n_steps
        # (n_episodes, n_steps, n_market_features), precomputed once
        self.market = batch.market_features()

    def observations(self, t: int, position: np.ndarray, avg_entry: np.ndarray,
                     steps_in: np.ndarray) -> np.ndarray:
        """assemble the (n_episodes, n_features) observation at step t.

        mirrors BinaryMarketEnv._obs, including the detail that the normalizer
        is applied to the market block ONLY and the position block passes
        through untouched.
        """
        market = self.market[:, t, :].copy()
        if self.normalizer is not None:
            market = self.normalizer.transform(market)

        mark = 0.5 * (self.batch.bid[:, t] + self.batch.ask[:, t])
        unrealized = np.where(position != 0.0, position * (mark - avg_entry), 0.0)

        pos_block = np.column_stack(
            [
                position / self.max_position,
                avg_entry,
                unrealized / self.max_position,
                np.minimum(steps_in / max(self.n_steps, 1), 1.0),
            ]
        )
        return np.concatenate([market, pos_block], axis=1).astype(np.float32)

    def run(self, policy: BatchPolicy) -> dict[str, np.ndarray]:
        """replay every episode. returns per-episode totals."""
        n = self.n_episodes
        cash = np.zeros(n)
        position = np.zeros(n)
        avg_entry = np.zeros(n)
        steps_in = np.zeros(n)
        trades = np.zeros(n)
        fees = np.zeros(n)

        for t in range(self.n_steps):
            obs = self.observations(t, position, avg_entry, steps_in)
            actions = np.asarray(policy(obs), dtype=int)
            target = np.asarray(ACTION_TO_TARGET, dtype=float)[actions] * self.max_position
            delta = target - position

            bid = self.batch.bid[:, t].astype(float)
            ask = self.batch.ask[:, t].astype(float)

            if self.costs_enabled:
                price = np.where(delta > 0, ask, bid)
                fee = _fee(delta, price)
            else:
                price = 0.5 * (bid + ask)
                fee = np.zeros(n)

            traded = delta != 0.0
            cash = np.where(traded, cash - delta * price - fee, cash)
            fees = np.where(traded, fees + fee, fees)
            trades = np.where(traded, trades + 1, trades)

            # average entry: reset to the fill price when opening or flipping,
            # blend when adding, unchanged when reducing. matches
            # PositionState.apply_fill.
            new_pos = position + delta
            same_sign = np.sign(delta) == np.sign(position)
            adding = traded & ((position == 0.0) | same_sign)
            total = np.abs(position) + np.abs(delta)
            blended = np.divide(
                np.abs(position) * avg_entry + np.abs(delta) * price,
                np.where(total == 0.0, 1.0, total),
            )
            reducing = traded & ~adding & (np.abs(delta) < np.abs(position))

            avg_entry = np.where(adding, blended, avg_entry)
            avg_entry = np.where(
                traded & ~adding & ~reducing,
                np.where(new_pos != 0.0, price, 0.0),
                avg_entry,
            )
            avg_entry = np.where(new_pos == 0.0, 0.0, avg_entry)

            flipped = traded & (np.sign(new_pos) != np.sign(position))
            steps_in = np.where(
                ~traded, steps_in + 1, np.where(flipped, 0.0, steps_in + 1)
            )
            steps_in = np.where(new_pos == 0.0, 0.0, steps_in)
            position = new_pos

        # terminal: mark the position at the true settlement value
        settlement = self.batch.settlement.astype(float)
        equity = cash + position * settlement

        return {
            "returns": equity,
            "trades": trades,
            "fees": fees,
            "final_position": position,
        }


def masked_policy(
    net_fn: Callable[[np.ndarray], np.ndarray],
    mask: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
) -> BatchPolicy:
    """wrap a policy so it only observes the features in `mask`.

    features outside the mask are replaced with draws from `background`, which
    is the interventional formulation used throughout this project. the draw is
    resampled at every step rather than fixed per episode: holding one draw
    constant would leak information through the *consistency* of the masked
    values, which is a subtle way for a masked agent to do better than it
    should.
    """

    def policy(obs: np.ndarray) -> np.ndarray:
        synthetic = obs.copy()
        idx = rng.integers(0, len(background), size=len(obs))
        draws = background[idx]
        synthetic[:, ~mask] = draws[:, ~mask]
        return net_fn(synthetic)

    return policy


def build_background(
    rollout: VectorizedRollout, n_samples: int = 512, seed: int = 0
) -> np.ndarray:
    """a reference distribution of observations, drawn from a flat agent.

    taken with the agent held flat so the position block reflects "no
    inventory", which is the natural reference point for asking what a feature
    contributed relative to doing nothing.
    """
    rng = np.random.default_rng(seed)
    n = rollout.n_episodes
    zeros = np.zeros(n)
    obs_all = []
    for t in range(rollout.n_steps):
        obs_all.append(rollout.observations(t, zeros, zeros, zeros))
    stacked = np.concatenate(obs_all, axis=0)
    idx = rng.choice(len(stacked), size=min(n_samples, len(stacked)), replace=False)
    return stacked[idx]


def greedy_policy(agent) -> Callable[[np.ndarray], np.ndarray]:
    """batched greedy action selection from a trained agent.

    torch is imported here rather than at module scope so that this module
    stays importable in the numpy-only paths that use VectorizedRollout without
    ever touching a network.
    """
    import torch

    def policy(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits, _ = agent.net(torch.as_tensor(obs, dtype=torch.float32))
            return logits.argmax(dim=-1).numpy()

    return policy


def masked_span(agent, rollout: "VectorizedRollout", background: np.ndarray,
                seed: int) -> float:
    """v(N) - v(empty) under `rollout`: the statistic the null test uses.

    lives here rather than in each script because four of them need it and an
    earlier divergence between two copies of a rollout is exactly the bug this
    project spent an evening finding. one definition, imported everywhere.

    the rng is created once and consumed only by the empty coalition, since the
    full one masks nothing. that makes the two evaluations deterministic given
    the seed, and comparable to each other.
    """
    rng = np.random.default_rng(seed)
    base = greedy_policy(agent)
    n_features = background.shape[1]
    full = np.ones(n_features, dtype=bool)
    empty = np.zeros(n_features, dtype=bool)

    def masked(mask: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def policy(obs: np.ndarray) -> np.ndarray:
            synthetic = obs.copy()
            if not mask.all():
                draws = background[rng.integers(0, len(background), size=len(obs))]
                synthetic[:, ~mask] = draws[:, ~mask]
            return base(synthetic)

        return policy

    return float(
        rollout.run(masked(full))["returns"].mean()
        - rollout.run(masked(empty))["returns"].mean()
    )
