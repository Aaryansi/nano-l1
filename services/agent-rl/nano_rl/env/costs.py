"""transaction-cost model for kalshi binary contracts.

two components, both measured rather than assumed:

  fees    kalshi's published taker formula, ceil(0.07 * C * P * (1-P)) rounded
          up to the next cent **per order**. verified against two independent
          public sources; see docs/MDP.md section 7.2.

  spread  crossing the quoted book. bid/ask come from the 1-minute candlestick
          containing the step, which is the only public source of quotes.

why this module matters more than it looks: the fee formula peaks at P = 0.5,
and the KXBTC15M contract sits near 0.5 by construction, so friction is at its
theoretical maximum for exactly the instrument being traded. getting this
wrong in the optimistic direction would manufacture false profit, so the
defaults here are deliberately conservative (pure taker, no maker rebate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# kalshi taker fee coefficient. fee = ceil_cents(TAKER_COEF * C * P * (1-P)).
TAKER_FEE_COEF = 0.07

# a maker formula with a smaller coefficient exists but is series-dependent and
# unconfirmed for KXBTC15M, so we do not model it. treating every fill as a
# taker is strictly conservative: it can only understate performance.
MAKER_FEE_COEF_UNCONFIRMED = 0.0175


def fee_dollars(n_contracts: float, price: float, coef: float = TAKER_FEE_COEF) -> float:
    """kalshi trading fee in dollars for one order.

    args:
        n_contracts: order size, contracts. sign is ignored; fees are charged
            on absolute size in both directions.
        price: contract price in dollars, in [0, 1].
        coef: fee coefficient. defaults to the taker rate.

    returns:
        fee in dollars, rounded up to the next whole cent, per order.

    note the rounding is per **order**, not per contract, which is why this
    takes the whole order size rather than being applied contract-wise. a
    per-contract ceiling would overstate fees by up to a cent per contract.
    """
    if n_contracts == 0:
        return 0.0

    raw = coef * abs(n_contracts) * price * (1.0 - price)
    # round up to the next cent. math.ceil on the cent-scaled value, guarding
    # against binary float representation nudging an exact cent upward.
    cents = math.ceil(round(raw * 100.0, 9))
    return cents / 100.0


def fee_per_contract(price: float, coef: float = TAKER_FEE_COEF) -> float:
    """unrounded per-contract fee, for analysis and plots.

    this is the quantity that peaks at 1.75 cents when price = 0.5 and the
    taker coefficient is used.
    """
    return coef * price * (1.0 - price)


@dataclass(frozen=True)
class Quote:
    """top of book for one decision step."""

    bid: float
    ask: float

    def __post_init__(self) -> None:
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: bid={self.bid} > ask={self.ask}")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class Fill:
    """the result of executing a position change."""

    n_contracts: float  # signed: positive = bought yes-equivalent
    price: float  # execution price actually paid/received, per contract
    fee: float  # dollars
    cash_delta: float  # signed change in cash, fees included


class CostModel:
    """converts a desired position change into an executed fill.

    execution assumption: the agent is a price taker and crosses the spread.
    a buy lifts the ask, a sell hits the bid. there is no queue simulation and
    no partial fill; at the sizes used here (single-digit contracts against a
    book with six-figure open interest) that is a reasonable simplification,
    and it is stated as such in the report.
    """

    def __init__(
        self,
        fee_coef: float = TAKER_FEE_COEF,
        impact_coef: float = 0.0,
        enabled: bool = True,
    ) -> None:
        """
        args:
            fee_coef: kalshi fee coefficient.
            impact_coef: linear market-impact coefficient, in price units per
                unit of (order size / bar volume). zero disables impact.
            enabled: when False, both fees and spread crossing are switched
                off and execution happens at mid. this exists for the zero-cost
                ablation described in docs/MDP.md section 9.4, which separates
                "cannot predict" from "predicts but cannot cover costs".
        """
        self.fee_coef = fee_coef
        self.impact_coef = impact_coef
        self.enabled = enabled

    def execute(
        self,
        delta_contracts: float,
        quote: Quote,
        bar_volume: float = 0.0,
    ) -> Fill:
        """execute a position change of `delta_contracts` against `quote`.

        args:
            delta_contracts: signed change in yes-equivalent position.
                positive buys, negative sells.
            quote: top of book at this step.
            bar_volume: contracts traded in this bar, used for the impact term.

        returns:
            a Fill describing price, fee, and the signed cash change.
        """
        if delta_contracts == 0:
            return Fill(0.0, quote.mid, 0.0, 0.0)

        if not self.enabled:
            # frictionless ablation: trade at mid, pay nothing.
            price = quote.mid
            return Fill(delta_contracts, price, 0.0, -delta_contracts * price)

        # cross the spread in the direction of the trade.
        price = quote.ask if delta_contracts > 0 else quote.bid

        # optional linear impact, pushing price further against the trader.
        if self.impact_coef > 0.0 and bar_volume > 0.0:
            participation = abs(delta_contracts) / bar_volume
            price += math.copysign(self.impact_coef * participation, delta_contracts)

        # a binary contract cannot trade outside [0, 1].
        price = min(1.0, max(0.0, price))

        fee = fee_dollars(delta_contracts, price, self.fee_coef)

        # buying costs cash and pays the fee; selling receives cash and still
        # pays the fee. hence the fee is always subtracted.
        cash_delta = -delta_contracts * price - fee

        return Fill(delta_contracts, price, fee, cash_delta)

    def round_trip_cost(self, price: float, n_contracts: float, spread: float) -> float:
        """total friction for opening and closing one position, in dollars.

        used for reporting the hurdle rate, not in the simulation loop.
        """
        fees = 2.0 * fee_dollars(n_contracts, price, self.fee_coef)
        spread_cost = abs(n_contracts) * spread
        return fees + spread_cost
