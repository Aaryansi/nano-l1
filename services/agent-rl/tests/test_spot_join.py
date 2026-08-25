"""tests for the binance spot join.

joining a second data source onto the kalshi clock is a new leak surface, and
the failure mode is quiet: a nearest-neighbour join looks correct in a plot and
silently hands the agent up to half a bar of the future. these tests pin the
join to backward as-of.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.data.binance import SpotSeries, _normalize_epoch, build_spot_features
from nano_rl.env.features import N_SPOT


@pytest.fixture
def ramp() -> SpotSeries:
    """price == timestamp, so any lookup error is immediately visible."""
    ts = np.arange(1000.0, 2000.0, 1.0)
    return SpotSeries(ts=ts, price=ts.copy())


class TestAsOf:
    def test_exact_hit_returns_that_bar(self, ramp) -> None:
        assert ramp.as_of(1500.0) == pytest.approx(1500.0)

    def test_between_bars_returns_the_earlier_one(self, ramp) -> None:
        """never round forward to the nearer bar."""
        assert ramp.as_of(1500.9) == pytest.approx(1500.0)

    def test_before_series_start_returns_none(self, ramp) -> None:
        assert ramp.as_of(999.0) is None

    def test_after_series_end_returns_last(self, ramp) -> None:
        assert ramp.as_of(5000.0) == pytest.approx(1999.0)

    def test_vectorised_matches_scalar(self, ramp) -> None:
        q = np.array([1000.0, 1234.5, 1500.0, 1999.0])
        vec = ramp.as_of_many(q)
        scalar = [ramp.as_of(x) for x in q]
        np.testing.assert_allclose(vec, scalar)

    def test_vectorised_is_nan_before_start(self, ramp) -> None:
        out = ramp.as_of_many(np.array([500.0, 1500.0]))
        assert np.isnan(out[0])
        assert out[1] == pytest.approx(1500.0)

    def test_never_returns_a_future_value(self, ramp) -> None:
        """the property that matters, checked across the whole series."""
        q = np.linspace(1000.0, 1999.0, 500)
        got = ramp.as_of_many(q)
        assert np.all(got <= q + 1e-9)


class TestEpochNormalisation:
    def test_milliseconds(self) -> None:
        # 2026-08-20T00:00:00Z in ms
        assert _normalize_epoch(1787270400000) == pytest.approx(1787270400.0)

    def test_microseconds(self) -> None:
        assert _normalize_epoch(1787270400000000) == pytest.approx(1787270400.0)

    def test_both_units_agree(self) -> None:
        """binance switched units mid-history; both must land on the same time."""
        ms = _normalize_epoch(1787270400000)
        us = _normalize_epoch(1787270400000000)
        assert ms == pytest.approx(us)


class TestSpotFeatures:
    def test_shape_and_finiteness(self, ramp) -> None:
        t_sec = np.arange(1, 11) * 60.0
        out = build_spot_features(
            ramp, open_epoch=1000.0, t_sec=t_sec, implied_prob=np.full(10, 0.5)
        )
        assert out.shape == (10, N_SPOT)
        assert np.all(np.isfinite(out))

    def test_return_since_open_is_measured_from_the_open(self, ramp) -> None:
        t_sec = np.array([60.0])
        out = build_spot_features(
            ramp, open_epoch=1000.0, t_sec=t_sec, implied_prob=np.array([0.5])
        )
        # price at 1060 vs reference at 1000
        assert out[0, 0] == pytest.approx(1060.0 / 1000.0 - 1.0)

    def test_missing_coverage_returns_zeros_not_garbage(self) -> None:
        """an episode outside the spot series must not fabricate a trend."""
        far = SpotSeries(ts=np.arange(1e9, 1e9 + 100), price=np.full(100, 50000.0))
        out = build_spot_features(
            far, open_epoch=1000.0, t_sec=np.arange(1, 6) * 60.0,
            implied_prob=np.full(5, 0.5),
        )
        assert out.shape == (5, N_SPOT)
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_future_spot_cannot_change_past_features(self) -> None:
        """the perturbation test, applied to the second data source."""
        ts = np.arange(0.0, 2000.0)
        px = np.full(2000, 50000.0)
        base = SpotSeries(ts=ts, price=px.copy())

        t_sec = np.arange(1, 11) * 60.0
        implied = np.full(10, 0.5)
        a = build_spot_features(base, 0.0, t_sec, implied)

        # corrupt spot strictly after the 5th decision boundary (t=300)
        px2 = px.copy()
        px2[int(300 + 1) :] = 90000.0
        b = build_spot_features(SpotSeries(ts=ts, price=px2), 0.0, t_sec, implied)

        np.testing.assert_array_equal(a[:5], b[:5])
        assert not np.array_equal(a[6:], b[6:])

    def test_gap_responds_to_disagreement(self) -> None:
        """the lead-lag feature must move when spot and implied disagree."""
        ts = np.arange(0.0, 2000.0)
        rising = SpotSeries(ts=ts, price=50000.0 * (1.0 + ts * 1e-6))
        t_sec = np.arange(1, 6) * 60.0

        # spot up but market still at even odds -> positive gap
        disagree = build_spot_features(rising, 0.0, t_sec, np.full(5, 0.5))
        # spot up and market already fully priced -> smaller gap
        agree = build_spot_features(rising, 0.0, t_sec, np.full(5, 0.95))

        assert disagree[-1, 4] > agree[-1, 4]
