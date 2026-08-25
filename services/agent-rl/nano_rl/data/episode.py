"""resample a kalshi market into a fixed-step, strictly causal episode.

this module is where no-lookahead is enforced, so the causality rule is stated
once and applied everywhere below:

    the observation at step boundary `t` may use only information that was
    publicly known at or before `t`.

two consequences that are easy to get wrong and are tested in
tests/test_no_lookahead.py:

  1. a 1-minute candle with `end_ts = T` summarises the interval (T-60, T]. it
     is therefore NOT observable until `T`. selecting the candle whose end_ts
     is nearest to `t`, or which contains `t`, leaks up to 59 seconds of the
     future. we select the latest candle with `end_ts <= t`.

  2. the settlement value is known only after `close_time`. it appears in the
     terminal transition and never as a feature.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from nano_rl.data.kalshi import Candle, Market, Trade


@dataclass(frozen=True)
class Episode:
    """one market resampled onto a fixed decision grid.

    every array has length `n_steps` and is aligned to `t_sec`, the seconds
    since market open at which the agent observes and acts.
    """

    ticker: str
    settlement: float  # 1.0 if resolved yes, else 0.0
    step_seconds: int

    t_sec: np.ndarray  # (n,) seconds since open at each decision boundary
    bid: np.ndarray  # (n,) best yes bid, last observable
    ask: np.ndarray  # (n,) best yes ask, last observable
    last_price: np.ndarray  # (n,) last traded yes price at or before t
    volume: np.ndarray  # (n,) contracts traded in the trailing bar
    staleness: np.ndarray  # (n,) seconds since the last observed trade
    flow_imbalance: np.ndarray  # (n,) signed taker imbalance, 0 when no tape

    has_tape: bool  # whether trade-level features are real or filled

    @property
    def n_steps(self) -> int:
        return len(self.t_sec)

    @property
    def mid(self) -> np.ndarray:
        """mark price used for mark-to-market accounting."""
        return 0.5 * (self.bid + self.ask)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """compact serialisable form, float32 to keep the corpus small."""
        return {
            "t_sec": self.t_sec.astype(np.float32),
            "bid": self.bid.astype(np.float32),
            "ask": self.ask.astype(np.float32),
            "last_price": self.last_price.astype(np.float32),
            "volume": self.volume.astype(np.float32),
            "staleness": self.staleness.astype(np.float32),
            "flow_imbalance": self.flow_imbalance.astype(np.float32),
        }


def _clean_candles(candles: list[Candle]) -> list[Candle]:
    """drop candles with no usable quote, keep ascending by end_ts."""
    usable = [c for c in candles if c.yes_bid_close is not None and c.yes_ask_close is not None]
    usable.sort(key=lambda c: c.end_ts)
    return usable


def build_episode(
    market: Market,
    candles: list[Candle],
    trades: list[Trade] | None = None,
    step_seconds: int = 60,
    imbalance_window_s: int = 60,
) -> Episode | None:
    """resample one market onto a `step_seconds` decision grid.

    args:
        market: the contract, providing episode bounds and settlement.
        candles: 1-minute bars, the only public source of bid/ask.
        trades: optional full tape. when supplied, enables `last_price` at
            sub-candle resolution plus real `flow_imbalance`. when omitted,
            `last_price` falls back to the candle close and `flow_imbalance`
            is zero.
        step_seconds: decision interval. must divide the 900s episode.
        imbalance_window_s: trailing window for the taker-flow imbalance.

    returns:
        an Episode, or None when the market has too little quote data to form
        a usable trajectory.

    the grid starts at `step_seconds` rather than 0, because at t=0 no candle
    has closed yet and there is therefore no observable quote. the final grid
    point is strictly before close, since the terminal transition is handled by
    the environment using the settlement value.
    """
    usable = _clean_candles(candles)
    if len(usable) < 2:
        return None

    duration = market.duration_s
    if duration <= 0 or step_seconds <= 0:
        return None

    # decision boundaries: step_seconds, 2*step, ..., strictly less than close.
    n_steps = int(duration // step_seconds) - 1
    if n_steps < 2:
        return None
    t_sec = np.arange(1, n_steps + 1, dtype=np.float64) * step_seconds

    # precompute sorted keys for causal (right-open) lookups.
    candle_ends = [(c.end_ts - market.open_time).total_seconds() for c in usable]

    trade_times: list[float] = []
    trade_px: list[float] = []
    trade_sz: list[float] = []
    trade_sign: list[float] = []
    if trades:
        for tr in trades:
            rel = (tr.ts - market.open_time).total_seconds()
            if rel < 0 or rel > duration:
                continue  # defensive: drop anything outside the market's life
            trade_times.append(rel)
            trade_px.append(tr.yes_price)
            trade_sz.append(tr.size)
            # taker buying yes lifts the offer, taker buying no presses the bid.
            trade_sign.append(1.0 if tr.taker_side == "yes" else -1.0)

    have_tape = len(trade_times) > 0
    px_arr = np.asarray(trade_px)
    sz_arr = np.asarray(trade_sz)
    sign_arr = np.asarray(trade_sign)
    cum_sz = np.concatenate([[0.0], np.cumsum(sz_arr)]) if have_tape else None
    cum_signed = np.concatenate([[0.0], np.cumsum(sz_arr * sign_arr)]) if have_tape else None

    bid = np.empty(n_steps)
    ask = np.empty(n_steps)
    last_price = np.empty(n_steps)
    volume = np.zeros(n_steps)
    staleness = np.full(n_steps, float(imbalance_window_s))
    imbalance = np.zeros(n_steps)

    for i, t in enumerate(t_sec):
        # ---- quote: latest candle that had already CLOSED at or before t.
        # bisect_right gives the count of ends <= t, so idx-1 is that candle.
        idx = bisect.bisect_right(candle_ends, t) - 1
        if idx < 0:
            # no candle closed yet; the grid should prevent this, but stay safe
            # by reusing the first available quote rather than inventing one.
            idx = 0
        c = usable[idx]
        # _clean_candles guarantees both sides are present on `usable`.
        b, a = float(c.yes_bid_close), float(c.yes_ask_close)  # type: ignore[arg-type]
        bid[i] = b
        ask[i] = a
        # a candle can carry quotes but no trade print, in which case the
        # quote mid is the best available mark.
        last_price[i] = c.close if c.close is not None else 0.5 * (b + a)
        volume[i] = c.volume

        if not have_tape:
            continue

        # ---- tape features: trades strictly at or before t.
        j = bisect.bisect_right(trade_times, t)
        if j > 0:
            last_price[i] = px_arr[j - 1]
            staleness[i] = t - trade_times[j - 1]
        else:
            staleness[i] = t  # nothing has traded yet this episode

        # trailing-window volume and signed imbalance via cumulative sums.
        k = bisect.bisect_right(trade_times, t - imbalance_window_s)
        win_sz = cum_sz[j] - cum_sz[k]  # type: ignore[index]
        win_signed = cum_signed[j] - cum_signed[k]  # type: ignore[index]
        volume[i] = win_sz
        imbalance[i] = (win_signed / win_sz) if win_sz > 0 else 0.0

    # a quote that is crossed or inverted is bad data; the cost model would
    # reject it downstream, so drop the episode rather than silently repair it.
    if np.any(ask < bid):
        return None

    return Episode(
        ticker=market.ticker,
        settlement=market.settlement_value,
        step_seconds=step_seconds,
        t_sec=t_sec,
        bid=bid,
        ask=ask,
        last_price=last_price,
        volume=volume,
        staleness=np.clip(staleness, 0.0, None),
        flow_imbalance=imbalance,
        has_tape=have_tape,
    )


def episode_end_time(market: Market, step_seconds: int) -> timedelta:
    """convenience for tests: wall-clock offset of the last decision boundary."""
    n = int(market.duration_s // step_seconds) - 1
    return timedelta(seconds=n * step_seconds)
