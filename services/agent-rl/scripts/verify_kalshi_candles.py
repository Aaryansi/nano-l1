"""probe kalshi candlesticks: do they carry bid/ask history (spread source)?"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"
UA = {"User-Agent": "nano-l1-research-spike/0.1"}


def get(path: str, **params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


ticker = "KXBTC15M-26AUG120030-30"
open_dt = datetime(2026, 8, 12, 4, 15, tzinfo=timezone.utc)
close_dt = datetime(2026, 8, 12, 4, 30, tzinfo=timezone.utc)
start, end = int(open_dt.timestamp()), int(close_dt.timestamp())

d = get(
    f"series/KXBTC15M/markets/{ticker}/candlesticks",
    start_ts=start,
    end_ts=end,
    period_interval=1,
)

cs = d.get("candlesticks", [])
print(f"candles returned: {len(cs)}")
if cs:
    print("\nfield structure of one candle:")
    print(json.dumps(cs[len(cs) // 2], indent=2)[:1200])
    print("\ntop-level keys:", list(d.keys()))

    # note: the api suffixes monetary fields with `_dollars` and size fields
    # with `_fp`. reading the unsuffixed names silently yields None, which is
    # how this probe initially appeared to return an empty series.
    print("\n--- per-minute walk (spread is the phase-2 friction input) ---")
    for c in cs:
        t = datetime.fromtimestamp(c["end_period_ts"], timezone.utc).strftime("%H:%M")
        pr = c.get("price", {})
        yb = c.get("yes_bid", {})
        ya = c.get("yes_ask", {})
        bid = yb.get("close_dollars")
        ask = ya.get("close_dollars")
        spread = (
            f"{float(ask) - float(bid):.4f}" if bid is not None and ask is not None else "n/a"
        )
        print(
            f"  {t}  o={pr.get('open_dollars')} c={pr.get('close_dollars')} "
            f"hi={pr.get('high_dollars')} lo={pr.get('low_dollars')} "
            f"| bid={bid} ask={ask} spread={spread} "
            f"| vol={c.get('volume_fp')} oi={c.get('open_interest_fp')}"
        )
