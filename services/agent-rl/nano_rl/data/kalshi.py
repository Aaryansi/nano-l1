"""kalshi public-api client for the KXBTC15M series.

read-only, unauthenticated. every endpoint used here is public: no api key or
account is required. responses are cached to disk per market so that a full
ingest is resumable and so that `reproduce.sh` does not re-hit the api.

api shape notes (verified against the live api, 2026-08-24):

  - monetary fields carry a `_dollars` suffix, size fields carry `_fp`.
    reading the unsuffixed name silently yields None.
  - `/markets` paginates via an opaque `cursor`; an empty batch or a missing
    cursor terminates.
  - `/markets/trades` caps `limit` at 1000, so a busy 15-minute market needs
    ~20 requests to drain.
  - candlesticks accept `period_interval` in **minutes** with a minimum of 1,
    and cap at 5000 candles per request.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"

# politeness: kalshi does not publish a hard public-read limit, so we self-limit.
# 0.15s between calls was stable across several thousand requests during the
# phase-1 spike.
DEFAULT_SLEEP_S = 0.15
MAX_RETRIES = 5


def _parse_ts(s: str) -> datetime:
    """parse an rfc-3339 timestamp into an aware utc datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Market:
    """one KXBTC15M contract, i.e. one rl episode."""

    ticker: str
    open_time: datetime
    close_time: datetime
    result: str  # "yes" | "no"
    settlement_value: float  # 1.0 | 0.0
    volume: float

    @property
    def duration_s(self) -> float:
        return (self.close_time - self.open_time).total_seconds()

    @property
    def settled(self) -> bool:
        return self.result in ("yes", "no")

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "Market":
        # settlement_value_dollars is absent on some rows even when `result` is
        # populated; derive it from `result` in that case rather than defaulting
        # to zero, which would silently corrupt every terminal reward.
        raw_settle = d.get("settlement_value_dollars")
        if raw_settle is not None:
            settlement = float(raw_settle)
        else:
            settlement = 1.0 if d.get("result") == "yes" else 0.0

        return cls(
            ticker=d["ticker"],
            open_time=_parse_ts(d["open_time"]),
            close_time=_parse_ts(d["close_time"]),
            result=str(d.get("result", "")),
            settlement_value=settlement,
            volume=float(d.get("volume_fp", 0) or 0),
        )


@dataclass(frozen=True)
class Trade:
    """one execution on the public tape."""

    ts: datetime
    yes_price: float
    size: float
    taker_side: str  # "yes" | "no"

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "Trade":
        return cls(
            ts=_parse_ts(d["created_time"]),
            yes_price=float(d["yes_price_dollars"]),
            size=float(d.get("count_fp", 0) or 0),
            taker_side=str(d.get("taker_side", "")),
        )


@dataclass(frozen=True)
class Candle:
    """one-minute ohlc bar, and the only source of bid/ask in the public api."""

    end_ts: datetime
    close: float | None
    yes_bid_close: float | None
    yes_ask_close: float | None
    volume: float

    @property
    def spread(self) -> float | None:
        """quoted spread, or None when the book was empty for the whole minute."""
        if self.yes_bid_close is None or self.yes_ask_close is None:
            return None
        return self.yes_ask_close - self.yes_bid_close

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "Candle":
        def money(block: str, field: str) -> float | None:
            v = d.get(block, {}).get(f"{field}_dollars")
            return float(v) if v is not None else None

        return cls(
            end_ts=datetime.fromtimestamp(d["end_period_ts"], timezone.utc),
            close=money("price", "close"),
            yes_bid_close=money("yes_bid", "close"),
            yes_ask_close=money("yes_ask", "close"),
            volume=float(d.get("volume_fp", 0) or 0),
        )


class KalshiClient:
    """cached, rate-limited, read-only client."""

    def __init__(
        self,
        cache_dir: str | Path,
        sleep_s: float = DEFAULT_SLEEP_S,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.sleep_s = sleep_s
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "nano-l1-agent-rl/0.1"})

        for sub in ("trades", "candles"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ http

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        """GET with bounded exponential backoff on 429 and 5xx."""
        url = f"{BASE_URL}/{path}"
        delay = self.sleep_s

        for attempt in range(MAX_RETRIES):
            resp = self._session.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                time.sleep(self.sleep_s)
                return resp.json()

            if resp.status_code == 429 or resp.status_code >= 500:
                delay = min(delay * 2, 8.0)
                log.warning(
                    "kalshi %s on %s, backing off %.1fs (attempt %d/%d)",
                    resp.status_code,
                    path,
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            resp.raise_for_status()

        raise RuntimeError(f"kalshi: {path} failed after {MAX_RETRIES} retries")

    def _paginate(self, path: str, key: str, **params: Any) -> Iterator[dict[str, Any]]:
        """walk an opaque-cursor endpoint until it is drained."""
        cursor: str | None = None
        while True:
            kw = dict(params)
            if cursor:
                kw["cursor"] = cursor
            page = self._get(path, **kw)
            batch = page.get(key) or []
            if not batch:
                return
            yield from batch
            cursor = page.get("cursor")
            if not cursor:
                return

    # --------------------------------------------------------------- markets

    def fetch_markets(self, max_markets: int | None = None, refresh: bool = False) -> list[Market]:
        """all settled KXBTC15M markets, newest first, cached as one json file.

        the api returns markets newest-first. we keep that order on disk and
        sort chronologically at the call site, so that taking the first N gives
        a contiguous recent window rather than a scattered sample.
        """
        cache = self.cache_dir / "markets.json"

        if cache.exists() and not refresh:
            raw = json.loads(cache.read_text())
            log.info("markets: %d from cache", len(raw))
        else:
            raw = []
            for m in self._paginate(
                "markets", "markets", series_ticker=SERIES, status="settled", limit=200
            ):
                raw.append(m)
                if max_markets and len(raw) >= max_markets:
                    break
            cache.write_text(json.dumps(raw))
            log.info("markets: %d fetched and cached", len(raw))

        markets = [Market.from_api(m) for m in raw]
        # drop anything that did not actually settle; a non-settled market has
        # no terminal reward and cannot form a valid episode.
        return [m for m in markets if m.settled]

    # ---------------------------------------------------------------- trades

    def fetch_trades(self, ticker: str, refresh: bool = False) -> list[Trade]:
        """full public trade tape for one market, cached per ticker.

        returned in ascending time order. the api yields newest-first, so this
        reverses; downstream feature code assumes ascending and would silently
        compute backwards momentum otherwise.
        """
        cache = self.cache_dir / "trades" / f"{ticker}.json"

        if cache.exists() and not refresh:
            raw = json.loads(cache.read_text())
        else:
            raw = list(self._paginate("markets/trades", "trades", ticker=ticker, limit=1000))
            cache.write_text(json.dumps(raw))

        trades = [Trade.from_api(t) for t in raw]
        trades.sort(key=lambda t: t.ts)
        return trades

    # ------------------------------------------------------------- candles

    def fetch_candles(self, market: Market, refresh: bool = False) -> list[Candle]:
        """one-minute candles spanning the market's life, cached per ticker.

        this is the only public source of bid/ask, so it is what makes the
        spread component of the cost model measured rather than assumed.
        """
        cache = self.cache_dir / "candles" / f"{market.ticker}.json"

        if cache.exists() and not refresh:
            raw = json.loads(cache.read_text())
        else:
            page = self._get(
                f"series/{SERIES}/markets/{market.ticker}/candlesticks",
                start_ts=int(market.open_time.timestamp()),
                end_ts=int(market.close_time.timestamp()),
                period_interval=1,
            )
            raw = page.get("candlesticks") or []
            cache.write_text(json.dumps(raw))

        candles = [Candle.from_api(c) for c in raw]
        candles.sort(key=lambda c: c.end_ts)
        return candles
