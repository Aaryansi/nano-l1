"""tests for the transaction-cost model.

the fee formula is the single most consequential number in this project (it
sets the ~9% round-trip hurdle that shapes every result), so it is tested
against hand-computed values rather than against itself.
"""

from __future__ import annotations

import math

import pytest

from nano_rl.env.costs import (
    TAKER_FEE_COEF,
    CostModel,
    Quote,
    fee_dollars,
    fee_per_contract,
)


class TestFeeFormula:
    """fee = ceil_cents(0.07 * C * P * (1-P))."""

    def test_peaks_at_half(self) -> None:
        """the fee is maximised at p=0.5, which is where this contract lives."""
        at_half = fee_per_contract(0.50)
        for p in (0.01, 0.1, 0.25, 0.4, 0.49, 0.51, 0.6, 0.75, 0.9, 0.99):
            assert fee_per_contract(p) < at_half, f"p={p} should be cheaper than 0.5"

    def test_per_contract_value_at_half(self) -> None:
        """0.07 * 0.5 * 0.5 = 0.0175 dollars = 1.75 cents."""
        assert fee_per_contract(0.50) == pytest.approx(0.0175)

    def test_symmetric_about_half(self) -> None:
        """p(1-p) is symmetric, so buying yes at p costs the same as at 1-p."""
        for p in (0.1, 0.2, 0.35, 0.45):
            assert fee_per_contract(p) == pytest.approx(fee_per_contract(1.0 - p))

    @pytest.mark.parametrize(
        "contracts,price,expected_cents",
        [
            # 0.07 * 100 * 0.5 * 0.5 = 1.75 dollars exactly -> 175 cents
            (100, 0.50, 175),
            # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> ceil to 2 cents
            (1, 0.50, 2),
            # 0.07 * 10 * 0.5 * 0.5 = 0.175 -> ceil to 18 cents
            (10, 0.50, 18),
            # 0.07 * 100 * 0.9 * 0.1 = 0.63 dollars -> 63 cents
            (100, 0.90, 63),
            # 0.07 * 100 * 0.1 * 0.9 = 0.63 -> same by symmetry
            (100, 0.10, 63),
        ],
    )
    def test_known_values(self, contracts: float, price: float, expected_cents: int) -> None:
        assert fee_dollars(contracts, price) == pytest.approx(expected_cents / 100.0)

    def test_rounds_up_never_down(self) -> None:
        """rounding is up to the next cent, per order."""
        # 0.07 * 1 * 0.5 * 0.5 = 0.0175, which must become 0.02 not 0.01
        assert fee_dollars(1, 0.50) == pytest.approx(0.02)

    def test_rounding_is_per_order_not_per_contract(self) -> None:
        """one order of 100 must cost less than 100 orders of 1.

        per-contract rounding would charge 100 * 2 cents = $2.00; the correct
        per-order rounding charges $1.75.
        """
        one_big = fee_dollars(100, 0.50)
        many_small = 100 * fee_dollars(1, 0.50)
        assert one_big == pytest.approx(1.75)
        assert many_small == pytest.approx(2.00)
        assert one_big < many_small

    def test_zero_size_is_free(self) -> None:
        assert fee_dollars(0, 0.5) == 0.0

    def test_sign_is_ignored(self) -> None:
        """fees are charged on absolute size, in both directions."""
        assert fee_dollars(-10, 0.5) == fee_dollars(10, 0.5)

    def test_extremes_are_cheap(self) -> None:
        """p(1-p) -> 0 at the boundaries, so a certain contract is nearly free."""
        assert fee_dollars(1, 0.0) == 0.0
        assert fee_dollars(1, 1.0) == 0.0


class TestRoundTripHurdle:
    """the number quoted in docs/MDP.md section 7.2 must be reproducible."""

    def test_nine_percent_hurdle_at_half(self) -> None:
        """round trip at p=0.5 with a 1-cent spread is ~9% of notional.

        this is the claim the whole project's framing rests on, so it is
        asserted rather than left in prose.
        """
        model = CostModel()
        n = 100
        price, spread = 0.50, 0.01

        cost = model.round_trip_cost(price=price, n_contracts=n, spread=spread)
        notional = n * price

        # fees: 2 * 1.75 = 3.50, spread: 100 * 0.01 = 1.00, total 4.50
        assert cost == pytest.approx(4.50)
        assert cost / notional == pytest.approx(0.09, abs=1e-9)

    def test_fees_dominate_spread(self) -> None:
        """fees are ~3.5x spread cost at p=0.5, per the corrected phase-1 figure."""
        model = CostModel()
        n, price, spread = 100, 0.50, 0.01
        fees = 2.0 * fee_dollars(n, price)
        spread_cost = n * spread
        assert fees / spread_cost == pytest.approx(3.5)
        assert model.round_trip_cost(price, n, spread) == pytest.approx(fees + spread_cost)


class TestQuote:
    def test_rejects_crossed_book(self) -> None:
        with pytest.raises(ValueError, match="crossed quote"):
            Quote(bid=0.60, ask=0.50)

    def test_mid_and_spread(self) -> None:
        q = Quote(bid=0.49, ask=0.51)
        assert q.mid == pytest.approx(0.50)
        assert q.spread == pytest.approx(0.02)

    def test_locked_book_is_allowed(self) -> None:
        """bid == ask is degenerate but not invalid."""
        q = Quote(bid=0.5, ask=0.5)
        assert q.spread == 0.0


class TestExecution:
    def test_buy_lifts_the_ask(self) -> None:
        model = CostModel()
        fill = model.execute(10, Quote(bid=0.49, ask=0.51))
        assert fill.price == pytest.approx(0.51)

    def test_sell_hits_the_bid(self) -> None:
        model = CostModel()
        fill = model.execute(-10, Quote(bid=0.49, ask=0.51))
        assert fill.price == pytest.approx(0.49)

    def test_buy_reduces_cash_by_notional_plus_fee(self) -> None:
        model = CostModel()
        q = Quote(bid=0.49, ask=0.51)
        fill = model.execute(10, q)
        expected_fee = fee_dollars(10, 0.51)
        assert fill.cash_delta == pytest.approx(-(10 * 0.51) - expected_fee)
        assert fill.fee == pytest.approx(expected_fee)

    def test_sell_increases_cash_but_still_pays_fee(self) -> None:
        """selling receives cash; the fee is subtracted in both directions."""
        model = CostModel()
        q = Quote(bid=0.49, ask=0.51)
        fill = model.execute(-10, q)
        expected_fee = fee_dollars(10, 0.49)
        assert fill.cash_delta == pytest.approx((10 * 0.49) - expected_fee)
        assert fill.fee > 0

    def test_no_trade_is_completely_free(self) -> None:
        """the target-position action space makes 'hold' a genuine no-op."""
        model = CostModel()
        fill = model.execute(0, Quote(bid=0.49, ask=0.51))
        assert fill.fee == 0.0
        assert fill.cash_delta == 0.0
        assert fill.n_contracts == 0.0

    def test_round_trip_loses_money_on_a_flat_market(self) -> None:
        """the core economic fact: buy then sell at an unchanged quote loses.

        this is what makes the agent's correct behaviour 'abstain'.
        """
        model = CostModel()
        q = Quote(bid=0.49, ask=0.51)
        buy = model.execute(10, q)
        sell = model.execute(-10, q)
        assert buy.cash_delta + sell.cash_delta < 0


class TestFrictionlessAblation:
    """the zero-cost mode must be genuinely free, or the ablation is meaningless."""

    def test_executes_at_mid(self) -> None:
        model = CostModel(enabled=False)
        q = Quote(bid=0.49, ask=0.51)
        assert model.execute(10, q).price == pytest.approx(0.50)
        assert model.execute(-10, q).price == pytest.approx(0.50)

    def test_charges_nothing(self) -> None:
        model = CostModel(enabled=False)
        fill = model.execute(10, Quote(bid=0.49, ask=0.51))
        assert fill.fee == 0.0

    def test_round_trip_is_exactly_flat(self) -> None:
        """with costs off, buy-then-sell at an unchanged quote nets to zero."""
        model = CostModel(enabled=False)
        q = Quote(bid=0.49, ask=0.51)
        buy = model.execute(10, q)
        sell = model.execute(-10, q)
        assert buy.cash_delta + sell.cash_delta == pytest.approx(0.0)


class TestMarketImpact:
    def test_impact_pushes_price_against_the_trader(self) -> None:
        model = CostModel(impact_coef=0.1)
        q = Quote(bid=0.49, ask=0.51)
        # participation = 50/100 = 0.5, impact = 0.1*0.5 = 0.05
        buy = model.execute(50, q, bar_volume=100)
        assert buy.price == pytest.approx(0.56)
        sell = model.execute(-50, q, bar_volume=100)
        assert sell.price == pytest.approx(0.44)

    def test_impact_scales_with_participation(self) -> None:
        model = CostModel(impact_coef=0.1)
        q = Quote(bid=0.49, ask=0.51)
        small = model.execute(1, q, bar_volume=1000)
        large = model.execute(100, q, bar_volume=1000)
        assert large.price > small.price

    def test_price_is_clamped_to_unit_interval(self) -> None:
        """a binary cannot trade outside [0,1] no matter how large the impact."""
        model = CostModel(impact_coef=10.0)
        q = Quote(bid=0.49, ask=0.51)
        assert model.execute(100, q, bar_volume=1).price <= 1.0
        assert model.execute(-100, q, bar_volume=1).price >= 0.0

    def test_no_impact_without_volume(self) -> None:
        """bar_volume=0 must not divide by zero."""
        model = CostModel(impact_coef=0.1)
        fill = model.execute(10, Quote(bid=0.49, ask=0.51), bar_volume=0)
        assert math.isfinite(fill.price)
        assert fill.price == pytest.approx(0.51)
