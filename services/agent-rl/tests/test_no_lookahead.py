"""no-lookahead tests.

this file is the one a reviewer should read first. everything else in the
project is worthless if these fail, because a backtest that can see the future
will manufacture arbitrary profit.

the central technique is a **future-perturbation test**: compute features,
then corrupt every data point strictly after step k, recompute, and assert the
features at steps <= k are bit-identical. any leakage path, however indirect,
changes those values and fails the assertion. this is stronger than inspecting
the code for forward indexing, because it catches leakage through aggregates,
normalisation, and library defaults too.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nano_rl.data.episode import build_episode
from nano_rl.data.kalshi import Candle, Market, Trade
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.features import FeatureNormalizer, build_market_features

from .conftest import make_batch

OPEN = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(seconds=900)


def make_market(settlement: float = 1.0) -> Market:
    return Market(
        ticker="TEST-1",
        open_time=OPEN,
        close_time=CLOSE,
        result="yes" if settlement else "no",
        settlement_value=settlement,
        volume=1000.0,
    )


def make_candles(closes: list[float], spread: float = 0.02) -> list[Candle]:
    """one candle per minute, `end_ts` at the END of the interval it covers."""
    return [
        Candle(
            end_ts=OPEN + timedelta(seconds=60 * (i + 1)),
            close=c,
            yes_bid_close=c - spread / 2,
            yes_ask_close=c + spread / 2,
            volume=100.0,
        )
        for i, c in enumerate(closes)
    ]


def make_trades(prices: list[float], every_s: int = 10) -> list[Trade]:
    return [
        Trade(
            ts=OPEN + timedelta(seconds=every_s * (i + 1)),
            yes_price=p,
            size=10.0,
            taker_side="yes" if i % 2 == 0 else "no",
        )
        for i, p in enumerate(prices)
    ]


class TestCandleCausality:
    """a candle ending at T summarises (T-60, T] and is unknown before T."""

    def test_quote_comes_from_a_closed_candle_only(self) -> None:
        # candle i closes at 60*(i+1) with a distinctive price
        closes = [0.10 * (i + 1) for i in range(15)]
        ep = build_episode(make_market(), make_candles(closes), step_seconds=60)
        assert ep is not None

        # at t=60 the only closed candle is the one ending at 60 -> close 0.10
        assert ep.last_price[0] == pytest.approx(0.10)
        # at t=120, the candle ending at 120 -> 0.20. using the candle that
        # *contains* t, or the nearest one, would give 0.30 here.
        assert ep.last_price[1] == pytest.approx(0.20)

    def test_future_candles_cannot_change_the_past(self) -> None:
        """the core perturbation test, at the resampler level."""
        closes = [0.5] * 15
        base = build_episode(make_market(), make_candles(closes), step_seconds=60)
        assert base is not None

        k = 5
        corrupted = list(closes)
        for i in range(k + 1, len(corrupted)):
            corrupted[i] = 0.99  # violently different future
        perturbed = build_episode(make_market(), make_candles(corrupted), step_seconds=60)
        assert perturbed is not None

        np.testing.assert_array_equal(base.bid[: k + 1], perturbed.bid[: k + 1])
        np.testing.assert_array_equal(base.ask[: k + 1], perturbed.ask[: k + 1])
        np.testing.assert_array_equal(
            base.last_price[: k + 1], perturbed.last_price[: k + 1]
        )
        # and the future genuinely did change, so the test is not vacuous
        assert not np.array_equal(base.last_price[k + 2 :], perturbed.last_price[k + 2 :])

    def test_grid_never_reaches_close(self) -> None:
        """the terminal transition belongs to the env, not the feature grid."""
        ep = build_episode(make_market(), make_candles([0.5] * 15), step_seconds=60)
        assert ep is not None
        assert ep.t_sec[-1] < 900


class TestTradeCausality:
    def test_future_trades_cannot_change_the_past(self) -> None:
        prices = [0.5] * 90
        candles = make_candles([0.5] * 15)
        base = build_episode(make_market(), candles, make_trades(prices), step_seconds=10)
        assert base is not None

        k = 30
        corrupted = list(prices)
        for i in range(k * 1 + 5, len(corrupted)):
            corrupted[i] = 0.01
        perturbed = build_episode(
            make_market(), candles, make_trades(corrupted), step_seconds=10
        )
        assert perturbed is not None

        np.testing.assert_array_equal(base.last_price[:k], perturbed.last_price[:k])
        np.testing.assert_array_equal(
            base.flow_imbalance[:k], perturbed.flow_imbalance[:k]
        )

    def test_trades_outside_the_window_are_dropped(self) -> None:
        """a trade stamped after close must never enter the episode."""
        candles = make_candles([0.5] * 15)
        good = make_trades([0.5] * 60)
        rogue = list(good) + [
            Trade(ts=CLOSE + timedelta(seconds=60), yes_price=0.99, size=1e9, taker_side="yes")
        ]
        a = build_episode(make_market(), candles, good, step_seconds=10)
        b = build_episode(make_market(), candles, rogue, step_seconds=10)
        assert a is not None and b is not None
        np.testing.assert_array_equal(a.last_price, b.last_price)
        np.testing.assert_array_equal(a.volume, b.volume)

    def test_staleness_reflects_only_past_trades(self) -> None:
        """with no trades yet, staleness must equal elapsed time, not zero."""
        candles = make_candles([0.5] * 15)
        # first trade only at t=300
        late = [
            Trade(ts=OPEN + timedelta(seconds=300), yes_price=0.6, size=5.0, taker_side="yes")
        ]
        ep = build_episode(make_market(), candles, late, step_seconds=60)
        assert ep is not None
        # step 0 is t=60, nothing has traded, so staleness == 60
        assert ep.staleness[0] == pytest.approx(60.0)
        # the pre-trade steps must not have borrowed the t=300 price
        assert ep.last_price[0] != pytest.approx(0.6)


class TestFeatureCausality:
    def test_differences_are_backward_only(self) -> None:
        """a forward or centred difference would leak one step of the future."""
        n = 10
        mid = np.arange(n, dtype=float) / 10.0
        feats = build_market_features(
            bid=mid - 0.01,
            ask=mid + 0.01,
            last_price=mid,
            volume=np.ones(n),
            staleness=np.zeros(n),
            flow_imbalance=np.zeros(n),
            t_sec=np.arange(1, n + 1) * 60.0,
            duration_s=(n + 1) * 60.0,
        )
        # index 2 is p_change_1; at step 0 there is no prior step, so it must
        # be exactly zero rather than a borrowed forward difference.
        assert feats[0, 2] == 0.0
        assert feats[1, 2] == pytest.approx(0.1)

    def test_future_perturbation_leaves_past_features_identical(self) -> None:
        n = 20
        mid = np.full(n, 0.5)
        kwargs = dict(
            volume=np.ones(n),
            staleness=np.zeros(n),
            flow_imbalance=np.zeros(n),
            t_sec=np.arange(1, n + 1) * 60.0,
            duration_s=(n + 1) * 60.0,
        )
        base = build_market_features(bid=mid - 0.01, ask=mid + 0.01, last_price=mid, **kwargs)

        k = 8
        future = mid.copy()
        future[k + 1 :] = 0.95
        perturbed = build_market_features(
            bid=future - 0.01, ask=future + 0.01, last_price=future, **kwargs
        )

        np.testing.assert_array_equal(base[: k + 1], perturbed[: k + 1])
        assert not np.array_equal(base[k + 2 :], perturbed[k + 2 :])

    def test_realized_vol_uses_a_trailing_window(self) -> None:
        n = 10
        mid = np.zeros(n)
        mid[5:] = 1.0  # a jump at index 5
        feats = build_market_features(
            bid=mid,
            ask=mid,
            last_price=mid,
            volume=np.ones(n),
            staleness=np.zeros(n),
            flow_imbalance=np.zeros(n),
            t_sec=np.arange(1, n + 1) * 60.0,
            duration_s=(n + 1) * 60.0,
        )
        # index 4 is p_realized_vol. steps before the jump must not see it.
        assert feats[4, 4] == pytest.approx(0.0)
        assert feats[5, 4] > 0.0


class TestSettlementIsNotAFeature:
    def test_observations_are_identical_regardless_of_outcome(self) -> None:
        """the resolution must be invisible until the terminal transition."""
        mid = np.full((1, 8), 0.5, dtype=np.float32)
        yes = make_batch(n_episodes=1, n_steps=8, mid_path=mid, settlement=[1.0])
        no = make_batch(n_episodes=1, n_steps=8, mid_path=mid, settlement=[0.0])

        env_y = BinaryMarketEnv(yes, random_episode_order=False)
        env_n = BinaryMarketEnv(no, random_episode_order=False)

        oy, _ = env_y.reset(seed=0, options={"episode": 0})
        on, _ = env_n.reset(seed=0, options={"episode": 0})
        np.testing.assert_array_equal(oy, on)

        # every pre-terminal observation must also match
        for _ in range(7):
            oy, _, ty, _, _ = env_y.step(1)
            on, _, tn, _, _ = env_n.step(1)
            assert ty == tn
            if not ty:
                np.testing.assert_array_equal(oy, on)

    def test_settlement_only_enters_the_terminal_reward(self) -> None:
        mid = np.full((1, 6), 0.5, dtype=np.float32)
        yes = make_batch(n_episodes=1, n_steps=6, mid_path=mid, settlement=[1.0])
        no = make_batch(n_episodes=1, n_steps=6, mid_path=mid, settlement=[0.0])

        ry, rn = [], []
        for batch, sink in ((yes, ry), (no, rn)):
            env = BinaryMarketEnv(batch, random_episode_order=False)
            env.reset(seed=0, options={"episode": 0})
            for _ in range(6):
                _, r, term, _, _ = env.step(2)  # LONG
                sink.append(r)
                if term:
                    break

        # identical until the last step, then they diverge
        np.testing.assert_allclose(ry[:-1], rn[:-1], atol=1e-12)
        assert ry[-1] != pytest.approx(rn[-1])


class TestNormalizerHygiene:
    def test_raises_if_used_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            FeatureNormalizer().transform(np.zeros((2, 3)))

    def test_fit_on_train_does_not_see_test(self) -> None:
        """stats must come from train only; test data must not move them."""
        train = np.random.default_rng(0).normal(0, 1, (100, 4))
        test = np.random.default_rng(1).normal(50, 10, (100, 4))

        n = FeatureNormalizer().fit(train)
        mean_before = n.mean.copy()
        n.transform(test)  # transforming must not update anything
        np.testing.assert_array_equal(n.mean, mean_before)

        # and the fitted stats must reflect train, not the pooled data
        np.testing.assert_allclose(n.mean, train.mean(axis=0))

    def test_zero_variance_column_does_not_produce_inf(self) -> None:
        x = np.ones((10, 3))
        out = FeatureNormalizer().fit(x).transform(x)
        assert np.all(np.isfinite(out))


class TestSplitHygiene:
    def test_subset_preserves_time_order(self) -> None:
        batch = make_batch(n_episodes=10, n_steps=5)
        idx = np.arange(3, 8)
        sub = batch.subset(idx)
        assert len(sub) == 5
        np.testing.assert_array_equal(sub.open_epoch, batch.open_epoch[idx])
        assert np.all(np.diff(sub.open_epoch) > 0)

    def test_splits_do_not_overlap(self) -> None:
        """contiguous, ordered, disjoint: the walk-forward invariant."""
        batch = make_batch(n_episodes=100, n_steps=5)
        order = np.argsort(batch.open_epoch)
        train, val, test = order[:60], order[60:80], order[80:]

        assert set(train).isdisjoint(val)
        assert set(val).isdisjoint(test)
        assert batch.open_epoch[train].max() < batch.open_epoch[val].min()
        assert batch.open_epoch[val].max() < batch.open_epoch[test].min()
