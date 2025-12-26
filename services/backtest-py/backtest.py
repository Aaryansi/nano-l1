#!/usr/bin/env python3
"""
Nano-L1 CSV backtester (standalone)

Offline backtest that replays a CSV of ticks and runs a tiny toy strategy.

Expected CSV format (header row required):

    ts,symbol,price,side,qty

- ts:     integer timestamp (ns / ms / whatever; we treat as opaque)
- symbol: string, e.g. "TEST"
- price:  float
- side:   "buy" or "sell"  (side of the *tick*, not our trade)
- qty:    float quantity

Strategy (simple demo, just for benchmarking + showing P&L):
-----------------------------------------------------------
- Track last trade price.
- If price ticks DOWN -> we buy `qty`.
- If price ticks UP   -> we sell `qty`.

We compute:
- cash (P&L ledger),
- position (net qty),
- mark-to-market P&L at last price,
- total ticks processed,
- elapsed wall-clock time,
- ticks/second.

Usage (host):
-------------
    python backtest.py --file /path/to/ticks.csv --max-ticks 2000000

In Docker (once wired into docker-compose):
-------------------------------------------
    docker compose run --rm backtest-py

You can override the CSV path via:
- CLI:  --file path
- or env: TICKS_CSV=/path/to/ticks.csv  (default inside container: /data/ticks.csv)
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass


@dataclass
class Tick:
    ts: int
    symbol: str
    price: float
    side: str
    qty: float


def parse_tick(row: dict) -> Tick:
    return Tick(
        ts=int(row["ts"]),
        symbol=row["symbol"],
        price=float(row["price"]),
        side=row["side"].strip().lower(),
        qty=float(row["qty"]),
    )


def backtest(csv_path: str, max_ticks: int | None = None) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ticks CSV not found at: {csv_path}")

    print(f"[backtest] loading from: {csv_path}")
    start = time.perf_counter()

    cash = 0.0         # + when we sell, - when we buy
    position = 0.0     # net quantity
    last_price = 0.0
    prev_price: float | None = None
    n_ticks = 0

    # Super simple "fade the last move" strategy
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tick = parse_tick(row)
            n_ticks += 1

            if prev_price is not None:
                if tick.price < prev_price:
                    # Buy qty
                    cash -= tick.price * tick.qty
                    position += tick.qty
                elif tick.price > prev_price:
                    # Sell qty
                    cash += tick.price * tick.qty
                    position -= tick.qty

            prev_price = tick.price
            last_price = tick.price

            if max_ticks is not None and n_ticks >= max_ticks:
                break

    end = time.perf_counter()
    elapsed = end - start if end > start else 0.0

    # Mark-to-market P&L
    mtm = cash + position * last_price if last_price > 0 else cash
    ticks_per_sec = n_ticks / elapsed if elapsed > 0 else 0.0

    print("\n[backtest] done")
    print(f"  ticks processed : {n_ticks:,}")
    print(f"  elapsed time    : {elapsed:.4f} s")
    print(f"  ticks / second  : {ticks_per_sec:,.0f} / s")
    print("")
    print("  final position  :", position)
    print("  last price      :", last_price)
    print("  cash PnL        :", round(cash, 4))
    print("  MTM PnL         :", round(mtm, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Nano-L1 CSV backtester")
    parser.add_argument(
        "--file",
        "-f",
        dest="file",
        default=None,
        help="Path to ticks CSV (default: env TICKS_CSV or /data/ticks.csv)",
    )
    parser.add_argument(
        "--max-ticks",
        "-n",
        dest="max_ticks",
        type=int,
        default=None,
        help="Optional max number of ticks to process (for quick tests)",
    )

    args = parser.parse_args()
    csv_path = args.file or os.environ.get("TICKS_CSV", "/data/ticks.csv")

    backtest(csv_path, max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
