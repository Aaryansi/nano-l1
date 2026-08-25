"""binance spot data for the underlying-price features.

the KXBTC15M contract resolves on the sign of the BTC spot move over its
window, so spot is not incidental context: it is the thing being predicted.
features 9-13 in docs/MDP.md come from here.

source choice: `data.binance.vision` publishes free daily bulk files with no
api key. we use **1-second klines** rather than the aggregated trade tape:

    1s klines     ~2.5 MB/day     enough resolution for a 60s decision grid
    aggTrades   ~353 MB/month     ~20x larger for detail we resample away

the join back onto the kalshi clock is **backward as-of**: the spot bar used
at kalshi time t is the last bar that had already CLOSED at or before t. a
nearest-neighbour or forward join would leak.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
import zipfile
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/spot/daily/klines"

# binance kline csv columns, no header row in the published files.
COL_OPEN_TIME = 0
COL_CLOSE = 4
COL_VOLUME = 5


@dataclass
class SpotSeries:
    """a 1-second spot price series, sorted ascending by timestamp.

    stored as parallel arrays rather than a dataframe: the only access pattern
    is a sorted-search as-of lookup, which is faster and clearer this way.
    """

    ts: np.ndarray  # (n,) epoch seconds, bar CLOSE time
    price: np.ndarray  # (n,) close price

    def __len__(self) -> int:
        return len(self.ts)

    def as_of(self, when: float) -> float | None:
        """last price known at or before `when`, or None if the series starts later.

        this is the only lookup in the module, and it is deliberately the only
        one: exposing an index-based accessor would make a forward read easy to
        write by accident.
        """
        i = bisect_right(self.ts, when)
        return float(self.price[i - 1]) if i > 0 else None

    def as_of_many(self, whens: np.ndarray) -> np.ndarray:
        """vectorised as_of. returns NaN where no prior bar exists."""
        idx = np.searchsorted(self.ts, whens, side="right") - 1
        out = np.where(idx >= 0, self.price[np.clip(idx, 0, None)], np.nan)
        return out


def _normalize_epoch(raw: int) -> float:
    """binance switched kline timestamps from ms to us in 2025.

    the published files are not self-describing, so infer the unit from
    magnitude rather than assuming. a millisecond timestamp for any plausible
    date is ~1e12; microseconds is ~1e15.
    """
    if raw > 1e14:
        return raw / 1_000_000.0
    return raw / 1_000.0


def download_day(
    symbol: str,
    day: date,
    cache_dir: str | Path,
    interval: str = "1s",
) -> Path | None:
    """download one daily kline zip, cached. returns None if not published."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}-{interval}-{day.isoformat()}.zip"
    dest = cache / name

    if dest.exists():
        return dest

    url = f"{BASE}/{symbol}/{interval}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    except Exception as exc:
        log.warning("binance %s unavailable: %s", name, str(exc)[:80])
        return None
    return dest


def load_day(path: Path) -> tuple[list[float], list[float]]:
    """parse one daily kline zip into (close_epoch_seconds, close_price)."""
    ts: list[float] = []
    px: list[float] = []

    with zipfile.ZipFile(path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            for row in csv.reader(text):
                if not row or not row[0].strip():
                    continue
                # some months ship a header row; skip anything non-numeric.
                try:
                    open_ms = int(row[COL_OPEN_TIME])
                    close_px = float(row[COL_CLOSE])
                except ValueError:
                    continue
                open_s = _normalize_epoch(open_ms)
                # a 1s bar opening at T closes at T+1 and is observable then.
                ts.append(open_s + 1.0)
                px.append(close_px)

    return ts, px


def load_range(
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: str | Path,
    interval: str = "1s",
) -> SpotSeries:
    """download and concatenate spot klines covering [start, end].

    missing days are skipped with a warning rather than raising: binance
    occasionally lags on the most recent day, and a partial series is still
    usable as long as the gap is visible in the join statistics.
    """
    ts_all: list[float] = []
    px_all: list[float] = []

    day = start.date()
    last = end.date()
    n_missing = 0

    while day <= last:
        p = download_day(symbol, day, cache_dir, interval)
        if p is None:
            n_missing += 1
        else:
            t, x = load_day(p)
            ts_all.extend(t)
            px_all.extend(x)
        day += timedelta(days=1)

    if not ts_all:
        raise RuntimeError(f"no binance data retrieved for {symbol} {start}..{end}")

    order = np.argsort(np.asarray(ts_all), kind="stable")
    series = SpotSeries(
        ts=np.asarray(ts_all)[order],
        price=np.asarray(px_all)[order],
    )

    span_d = (series.ts[-1] - series.ts[0]) / 86400
    log.info(
        "binance %s: %d bars, %.1f days, %d missing days",
        symbol,
        len(series),
        span_d,
        n_missing,
    )
    return series


def build_spot_features(
    spot: SpotSeries,
    open_epoch: float,
    t_sec: np.ndarray,
    implied_prob: np.ndarray,
    lookbacks: tuple[int, int] = (30, 60),
) -> np.ndarray:
    """spot-derived features for one episode, aligned to its decision grid.

    args:
        spot: the underlying series.
        open_epoch: episode open time, epoch seconds. this is the contract's
            reference level: it resolves yes if spot at close exceeds spot here.
        t_sec: decision boundaries, seconds since open.
        implied_prob: the kalshi mid at each boundary, for the lead-lag gap.
        lookbacks: short-horizon return windows, in seconds.

    returns:
        (n_steps, 5) array: spot_ret_since_open, spot_ret_a, spot_ret_b,
        spot_realized_vol, spot_implied_gap.

    every lookup is `as_of`, i.e. backward. the reference price is taken at the
    episode open, which is known at t=0 and is not future information.
    """
    n = len(t_sec)
    abs_t = open_epoch + t_sec

    ref = spot.as_of(open_epoch)
    now = spot.as_of_many(abs_t)

    if ref is None or not np.isfinite(now).any():
        # no spot coverage for this episode; return zeros and let the caller
        # decide whether to drop it. silently imputing a trend would be worse.
        return np.zeros((n, 5))

    ret_since_open = (now / ref) - 1.0

    lagged = [spot.as_of_many(abs_t - lb) for lb in lookbacks]
    ret_a = (now / lagged[0]) - 1.0
    ret_b = (now / lagged[1]) - 1.0

    # trailing realised vol of the short-horizon returns, causal by
    # construction since ret_a itself only looks backward.
    vol = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 3 + 1)
        w = ret_a[lo : i + 1]
        vol[i] = np.nanstd(w) if len(w) > 1 else 0.0

    # lead-lag: how far the spot move has gone versus how far the kalshi price
    # has moved from even odds. the scale factor is arbitrary but fixed; the
    # agent can learn its own weighting.
    gap = np.tanh(ret_since_open * 500.0) - 2.0 * (implied_prob - 0.5)

    out = np.column_stack([ret_since_open, ret_a, ret_b, vol, gap])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
