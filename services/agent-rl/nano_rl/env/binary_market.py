"""gymnasium environment for trading kalshi binary contracts.

one episode is one KXBTC15M contract. the agent picks a target inventory at
each decision step and is paid the change in mark-to-market equity net of
transaction costs.

three accounting properties this env guarantees, each asserted in
tests/test_env_accounting.py:

  1. rewards telescope. sum(rewards) == final equity - initial equity, exactly.
     this is what makes the dense per-step reward return-equivalent to the
     sparse terminal reward at gamma=1 rather than a shaping heuristic.

  2. the terminal position marks at the true settlement value in {0, 1}, never
     at the last traded price.

  3. settlement is free. kalshi does not charge a fee to hold a contract to
     expiry, so holding to settlement avoids one leg of the round-trip fee.
     this is economically real and materially changes the optimal policy, so
     it is modelled rather than abstracted away. set
     `force_close_at_last_quote=True` to liquidate into the final market
     quote instead and measure how much that exemption is worth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # keep the module importable for pure-accounting tests
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from nano_rl.env.costs import CostModel, Quote
from nano_rl.env.features import (
    N_FEATURES,
    N_SPOT,
    FeatureNormalizer,
    PositionState,
    build_market_features,
)

# action index -> target position as a fraction of max_position.
ACTION_TO_TARGET: tuple[float, ...] = (-1.0, 0.0, 1.0)
ACTION_NAMES: tuple[str, ...] = ("SHORT", "FLAT", "LONG")


@dataclass
class EpisodeBatch:
    """a stacked corpus of episodes, as written by scripts/build_corpus.py.

    every array is (n_episodes, n_steps) except `settlement` and `open_epoch`,
    which are (n_episodes,).
    """

    bid: np.ndarray
    ask: np.ndarray
    last_price: np.ndarray
    volume: np.ndarray
    staleness: np.ndarray
    flow_imbalance: np.ndarray
    t_sec: np.ndarray
    settlement: np.ndarray
    open_epoch: np.ndarray

    # (n_episodes, n_steps, N_SPOT). None when the corpus was built without
    # binance coverage, in which case the block is zero-filled at use time so
    # that observation width stays fixed at N_FEATURES.
    spot: np.ndarray | None = None

    # cached per-episode market features, built lazily on first use
    _market_features: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: str) -> "EpisodeBatch":
        d = np.load(path)
        return cls(
            bid=d["bid"],
            ask=d["ask"],
            last_price=d["last_price"],
            volume=d["volume"],
            staleness=d["staleness"],
            flow_imbalance=d["flow_imbalance"],
            t_sec=d["t_sec"],
            settlement=d["settlement"],
            open_epoch=d["open_epoch"],
            spot=d["spot"] if "spot" in d.files else None,
        )

    @property
    def has_spot(self) -> bool:
        return self.spot is not None

    def __len__(self) -> int:
        return len(self.settlement)

    @property
    def n_steps(self) -> int:
        return self.bid.shape[1]

    @property
    def duration_s(self) -> float:
        """total episode length, inferred from the decision grid."""
        step = float(self.t_sec[0, 1] - self.t_sec[0, 0])
        return float(self.t_sec[0, -1] + step)

    def market_features(self) -> np.ndarray:
        """(n_episodes, n_steps, len(MARKET_FEATURES)), computed once.

        concatenates the kalshi block with the spot block. when the corpus has
        no spot coverage the spot block is zeros, which keeps the observation
        width fixed and makes the absence visible in attribution (a zero
        feature gets zero Shapley value) rather than silently changing shape.
        """
        if self._market_features is None:
            kalshi = np.stack(
                [
                    build_market_features(
                        bid=self.bid[i],
                        ask=self.ask[i],
                        last_price=self.last_price[i],
                        volume=self.volume[i],
                        staleness=self.staleness[i],
                        flow_imbalance=self.flow_imbalance[i],
                        t_sec=self.t_sec[i],
                        duration_s=self.duration_s,
                    )
                    for i in range(len(self))
                ]
            )
            spot = (
                self.spot
                if self.spot is not None
                else np.zeros((len(self), self.n_steps, N_SPOT), dtype=kalshi.dtype)
            )
            self._market_features = np.concatenate([kalshi, spot], axis=-1)
        return self._market_features

    def subset(self, idx: np.ndarray) -> "EpisodeBatch":
        """a time-ordered slice, used to build train/val/test splits."""
        return EpisodeBatch(
            bid=self.bid[idx],
            ask=self.ask[idx],
            last_price=self.last_price[idx],
            volume=self.volume[idx],
            staleness=self.staleness[idx],
            flow_imbalance=self.flow_imbalance[idx],
            t_sec=self.t_sec[idx],
            settlement=self.settlement[idx],
            open_epoch=self.open_epoch[idx],
            spot=self.spot[idx] if self.spot is not None else None,
        )


class BinaryMarketEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """target-position trading on a binary contract.

    observation: `N_FEATURES` floats, normalised if a fitted normalizer is
        supplied. see nano_rl/env/features.py for the layout.
    action: Discrete(3), mapping to target inventory {-max, 0, +max}.
    reward: change in mark-to-market equity, net of fees and spread, in
        dollars.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        batch: EpisodeBatch,
        cost_model: CostModel | None = None,
        normalizer: FeatureNormalizer | None = None,
        max_position: float = 100.0,
        force_close_at_last_quote: bool = False,
        random_episode_order: bool = True,
    ) -> None:
        """
        args:
            batch: the corpus this env samples episodes from.
            cost_model: friction model. defaults to the kalshi taker model.
            normalizer: fitted on TRAIN ONLY. None leaves features raw.
            max_position: inventory cap in contracts. 100 is the default
                because the fee rounds up per order to the next cent, so very
                small orders pay a disproportionate rounding penalty that is an
                artefact of size rather than of strategy.
            force_close_at_last_quote: when True, any open position is
                liquidated into the final market quote instead of being
                held to expiry. holding to expiry is free on kalshi, so
                this is strictly more expensive and exists to measure how
                much of the agent's pnl depends on that exemption.
            random_episode_order: shuffle episode order each pass. training
                wants this on; evaluation wants it off for reproducibility.
        """
        self.batch = batch
        self.cost_model = cost_model or CostModel()
        self.normalizer = normalizer
        self.max_position = max_position
        self.force_close_at_last_quote = force_close_at_last_quote
        self.random_episode_order = random_episode_order

        self.n_steps = batch.n_steps
        self._features = batch.market_features()

        if spaces is not None:
            self.action_space = spaces.Discrete(len(ACTION_TO_TARGET))
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
            )

        if normalizer is None:
            # a tanh trunk saturates on large-magnitude inputs, and the
            # resulting zero gradients look exactly like "the agent did not
            # learn" rather than like a bug. warn loudly rather than let a
            # silent 31%-saturated network be mistaken for a negative result.
            # see scripts/diagnose_ppo.py.
            peak = float(np.abs(self._features).max())
            if peak > 5.0:
                warnings.warn(
                    f"BinaryMarketEnv built without a normalizer and features "
                    f"reach magnitude {peak:.1f}. neural agents will train "
                    f"poorly. fit one with nano_rl.data.splits.walk_forward_split "
                    f"(real data) or nano_rl.env.features.fit_normalizer "
                    f"(synthetic).",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._rng = np.random.default_rng(0)
        self._order: np.ndarray = np.arange(len(batch))
        self._order_pos = 0

        # per-episode state
        self._ep = 0
        self._t = 0
        self._cash = 0.0
        self._pos = PositionState()
        self._prev_equity = 0.0
        self._trades = 0
        self._fees = 0.0

    # ------------------------------------------------------------ gym api

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._order = np.arange(len(self.batch))
            if self.random_episode_order:
                self._rng.shuffle(self._order)
            self._order_pos = 0

        if options and "episode" in options:
            self._ep = int(options["episode"])
        else:
            if self._order_pos >= len(self._order):
                self._order_pos = 0
                if self.random_episode_order:
                    self._rng.shuffle(self._order)
            self._ep = int(self._order[self._order_pos])
            self._order_pos += 1

        self._t = 0
        self._cash = 0.0
        self._pos = PositionState()
        self._trades = 0
        self._fees = 0.0
        # initial equity is zero: no cash, no inventory. this makes
        # sum(rewards) equal final equity exactly.
        self._prev_equity = 0.0

        return self._obs(), {"episode": self._ep, "ticker_idx": self._ep}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = ACTION_TO_TARGET[int(action)] * self.max_position
        delta = target - self._pos.position

        quote = Quote(bid=float(self.batch.bid[self._ep, self._t]),
                      ask=float(self.batch.ask[self._ep, self._t]))
        fill = self.cost_model.execute(
            delta, quote, bar_volume=float(self.batch.volume[self._ep, self._t])
        )

        if fill.n_contracts != 0.0:
            self._cash += fill.cash_delta
            self._pos.apply_fill(fill.n_contracts, fill.price)
            self._trades += 1
            self._fees += fill.fee
        else:
            self._pos.apply_fill(0.0, quote.mid)

        # advance the clock
        self._t += 1
        terminated = self._t >= self.n_steps

        if terminated:
            if self.force_close_at_last_quote and self._pos.position != 0.0:
                # liquidate into the last observable market quote rather than
                # letting the contract expire.
                #
                # note this must use the market quote, NOT the settlement
                # price. closing at settlement is economically identical to
                # expiring, and it is also literally free, because the fee
                # carries a P*(1-P) term that vanishes at P in {0, 1}. an
                # earlier version of this branch charged at the settlement
                # mark and was therefore a silent no-op.
                last = self.n_steps - 1
                closing = self.cost_model.execute(
                    -self._pos.position,
                    Quote(
                        bid=float(self.batch.bid[self._ep, last]),
                        ask=float(self.batch.ask[self._ep, last]),
                    ),
                    bar_volume=float(self.batch.volume[self._ep, last]),
                )
                self._cash += closing.cash_delta
                self._fees += closing.fee
                self._pos.apply_fill(closing.n_contracts, closing.price)

            mark = float(self.batch.settlement[self._ep])
        else:
            mark = float(0.5 * (self.batch.bid[self._ep, self._t]
                                + self.batch.ask[self._ep, self._t]))

        equity = self._cash + self._pos.position * mark
        reward = equity - self._prev_equity
        self._prev_equity = equity

        info = {
            "equity": equity,
            "cash": self._cash,
            "position": self._pos.position,
            "fees": self._fees,
            "trades": self._trades,
            "mark": mark,
        }
        return self._obs(), float(reward), terminated, False, info

    # -------------------------------------------------------------- helpers

    def _obs(self) -> np.ndarray:
        """observation at the current step.

        at the terminal boundary there is no further market row, so we reuse
        the last one for shape stability. this value is never acted upon: the
        env has already returned terminated=True.
        """
        t = min(self._t, self.n_steps - 1)
        market = self._features[self._ep, t]
        mark = float(0.5 * (self.batch.bid[self._ep, t] + self.batch.ask[self._ep, t]))
        pos = self._pos.features(mark, self.max_position, self.n_steps)

        # the normalizer is fit on the MARKET block only, so it must be applied
        # to that block only. position features are already on a unit scale by
        # construction (position/max_position, price in [0,1], pnl/max_position,
        # steps/n_steps) and would be corrupted by market-fitted statistics.
        if self.normalizer is not None:
            market = self.normalizer.transform(market)

        return np.concatenate([market, pos]).astype(np.float32)
