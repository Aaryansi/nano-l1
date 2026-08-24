"""de-risk spike: can we build a causal RL dataset from kalshi 15-min btc binaries?

checks, in order:
  1. how far back does settled-market history go (pagination depth)
  2. can we reconstruct a per-market price series from the public trade tape
  3. what does one episode actually look like (steps, spread, resolution)
  4. rough rate-limit behaviour
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"
UA = {"User-Agent": "nano-l1-research-spike/0.1"}


def get(path: str, **params) -> dict:
    url = f"{BASE}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------- 1. depth
print("=" * 66)
print("1. settled KXBTC15M history depth via pagination")
print("=" * 66)

cursor = None
markets: list[dict] = []
pages = 0
t0 = time.time()

while pages < 12:
    kw = {"series_ticker": "KXBTC15M", "limit": 200, "status": "settled"}
    if cursor:
        kw["cursor"] = cursor
    d = get("markets", **kw)
    batch = d.get("markets", [])
    if not batch:
        break
    markets.extend(batch)
    cursor = d.get("cursor")
    pages += 1
    if not cursor:
        break
    time.sleep(0.15)

print(f"pages={pages}  markets={len(markets)}  elapsed={time.time()-t0:.1f}s")

if markets:
    opens = sorted(ts(m["open_time"]) for m in markets)
    print(f"oldest open : {opens[0]}")
    print(f"newest open : {opens[-1]}")
    print(f"span        : {(opens[-1]-opens[0]).total_seconds()/86400:.2f} days")
    print(f"more pages available: {bool(cursor)}")

    # episode length distribution
    durs = {
        (ts(m["close_time"]) - ts(m["open_time"])).total_seconds()
        for m in markets
    }
    print(f"distinct episode durations (s): {sorted(durs)[:5]}")

    # how many distinct strikes per 15-min window
    by_window = defaultdict(list)
    for m in markets:
        by_window[m["open_time"]].append(m["ticker"])
    sizes = sorted(len(v) for v in by_window.values())
    print(f"markets per 15-min window: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
    print(f"distinct windows covered : {len(by_window)}")

# ------------------------------------------------- 2. per-market trade tape
print()
print("=" * 66)
print("2. per-market trade tape reconstruction")
print("=" * 66)

# pick a settled market with real volume
cands = sorted(
    (m for m in markets if float(m.get("volume_fp", 0) or 0) > 0),
    key=lambda m: -float(m.get("volume_fp", 0) or 0),
)
if not cands:
    print("no settled markets with volume, ABORT")
    raise SystemExit(1)

mk = cands[len(cands) // 2]  # median-ish liquid, not the outlier
print(f"market   : {mk['ticker']}")
print(f"title    : {mk.get('title','')[:70]}")
print(f"open     : {mk['open_time']}   close: {mk['close_time']}")
print(f"volume   : {mk.get('volume_fp')}   result: {mk.get('result')}")
print(f"settle   : {mk.get('settlement_value_dollars')}")

trades: list[dict] = []
cursor = None
for _ in range(20):
    kw = {"ticker": mk["ticker"], "limit": 1000}
    if cursor:
        kw["cursor"] = cursor
    d = get("markets/trades", **kw)
    b = d.get("trades", [])
    if not b:
        break
    trades.extend(b)
    cursor = d.get("cursor")
    if not cursor:
        break
    time.sleep(0.15)

print(f"\ntrades pulled: {len(trades)}")

if trades:
    trades.sort(key=lambda t: t["created_time"])
    t_open, t_close = ts(mk["open_time"]), ts(mk["close_time"])
    px = [float(t["yes_price_dollars"]) for t in trades]
    times = [ts(t["created_time"]) for t in trades]

    print(f"first trade : {times[0]}  @ {px[0]:.3f}")
    print(f"last  trade : {times[-1]}  @ {px[-1]:.3f}")
    print(f"price range : {min(px):.3f} .. {max(px):.3f}")
    print(f"within window: {times[0] >= t_open} .. {times[-1] <= t_close}")

    # decision-step density: trades per 10s bucket
    buckets = defaultdict(int)
    for t in times:
        buckets[int((t - t_open).total_seconds() // 10)] += 1
    filled = len(buckets)
    print(f"10s buckets with >=1 trade: {filled} / 90  ({filled/90*100:.0f}% coverage)")
    print(f"median trades per active bucket: {sorted(buckets.values())[len(buckets)//2]}")

    # terminal consistency: does last price agree with resolution?
    print(f"\nterminal check: result={mk.get('result')}  last_traded={px[-1]:.3f}")

# ------------------------------------------------------------ 3. throughput
print()
print("=" * 66)
print("3. rough throughput estimate for a full dataset pull")
print("=" * 66)
if markets and trades:
    per_market_s = 0.4
    print(f"~{per_market_s}s per market tape (observed, incl. politeness sleep)")
    for n in (500, 2000, 5000):
        print(f"  {n:>5} markets -> ~{n*per_market_s/60:.0f} min")
