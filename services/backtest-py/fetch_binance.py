#!/usr/bin/env python
import argparse
import csv
import time
from pathlib import Path

import requests


BASE_URL = "https://api.binance.us/api/v3/aggTrades"
MAX_PER_CALL = 1000


def fetch_agg_trades(symbol: str, limit: int, out_path: Path, hours_back: int = 24):
    """
    Fetch up to `limit` aggregated trades for `symbol` from Binance.

    Strategy:
      - Start `hours_back` hours in the past.
      - Call /aggTrades with startTime, limit=1000.
      - For each batch, move startTime to last_trade_ts + 1 ms.
      - Stop when:
          * we collected `limit` trades, or
          * Binance returns an empty batch.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[binance] fetching up to {limit} trades for {symbol}")
    print(f"[binance] looking back ~{hours_back}h from now")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - hours_back * 60 * 60 * 1000

    all_rows = []
    total = 0

    while total < limit:
        remaining = min(MAX_PER_CALL, limit - total)
        params = {
            "symbol": symbol,
            "limit": remaining,
            "startTime": start_ms,
        }

        resp = requests.get(BASE_URL, params=params, timeout=10)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"[binance] HTTP error: {e} (status={resp.status_code})")
            print(f"[binance] response: {resp.text[:300]}...")
            break

        batch = resp.json()
        if not batch:
            print("[binance] no more trades returned, stopping.")
            break

        # Convert to a simple CSV schema
        for t in batch:
            # Binance aggTrade fields:
            # a: aggId, p: price, q: qty, T: timestamp, m: isBuyerMaker (bool)
            side = "sell" if t.get("m", False) else "buy"
            all_rows.append(
                {
                    "ts": t["T"],
                    "symbol": symbol,
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "side": side,
                    "aggId": t["a"],
                }
            )

        total += len(batch)
        last_ts = batch[-1]["T"]
        start_ms = last_ts + 1  # next call starts after last trade timestamp

        print(
            f"[binance] fetched batch: {len(batch)} "
            f"(total={total}, last aggId={batch[-1]['a']}, last ts={last_ts})"
        )

        # small sleep to be polite
        time.sleep(0.2)

    if not all_rows:
        print("[binance] no data collected, giving up.")
        return

    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ts", "symbol", "price", "qty", "side", "aggId"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"[binance] wrote {len(all_rows)} trades to {out_path} "
        f"({out_path.resolve()})"
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance agg trades to CSV")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help="max number of trades to fetch (default 50k)",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=24,
        help="how many hours back from now to start fetching (default 24)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output CSV path (relative or absolute)",
    )
    args = parser.parse_args()

    fetch_agg_trades(args.symbol, args.limit, Path(args.out), hours_back=args.hours_back)


if __name__ == "__main__":
    main()
