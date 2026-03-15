import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Iterator, Optional


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass
class Tick:
    ts: int
    symbol: str
    price: float
    side: str   # "buy" | "sell"
    qty: float


# ---------------------------------------------------------------------
# Tick sources
# ---------------------------------------------------------------------

def iter_csv_ticks(path: str, max_ticks: Optional[int] = None) -> Iterator[Tick]:
    """stream ticks from csv (ts,symbol,price,side,qty)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ticks CSV not found at: {path}")

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            try:
                ts = int(row["ts"])
                symbol = row.get("symbol", "TEST")
                price = float(row["price"])
                side = row["side"].strip().lower()
                qty = float(row["qty"])
            except Exception as e:
                # skip bad rows
                print(f"[backtest] skipping bad row: {row} (err={e})")
                continue

            yield Tick(ts=ts, symbol=symbol, price=price, side=side, qty=qty)
            count += 1
            if max_ticks is not None and count >= max_ticks:
                break


def iter_synthetic_ticks(
    n: int,
    symbol: str = "TEST",
    start_price: float = 100.0,
    vol: float = 0.02,
) -> Iterator[Tick]:
    """random walk price generator for benchmarks"""
    price = start_price
    ts = 0

    for _ in range(n):
        ts += 1

        # Gaussian bump around 0; clamp to > 0
        bump = random.gauss(0.0, vol)
        price = max(0.01, price + bump)

        side = "buy" if random.random() < 0.5 else "sell"
        qty = random.randint(1, 5)

        yield Tick(ts=ts, symbol=symbol, price=price, side=side, qty=qty)


# ---------------------------------------------------------------------
# Strategy & PnL
# ---------------------------------------------------------------------

@dataclass
class BacktestResult:
    ticks_processed: int
    elapsed_sec: float
    final_position: float
    last_price: float
    cash_pnl: float
    mtm_pnl: float


def run_strategy(ticks: Iterator[Tick], max_ticks: Optional[int] = None) -> BacktestResult:
    """takes opposite side of each tick - simple market making sim"""
    position = 0.0
    cash = 0.0
    last_price = 0.0
    count = 0

    t0 = time.perf_counter()

    for t in ticks:
        price = t.price
        qty = t.qty
        last_price = price

        if t.side == "buy":
            # market wants to buy; we sell
            position -= qty
            cash += price * qty
        else:
            # market wants to sell; we buy
            position += qty
            cash -= price * qty

        count += 1
        if max_ticks is not None and count >= max_ticks:
            break

    t1 = time.perf_counter()
    elapsed = t1 - t0

    mtm = position * last_price if count > 0 else 0.0

    return BacktestResult(
        ticks_processed=count,
        elapsed_sec=elapsed,
        final_position=position,
        last_price=last_price,
        cash_pnl=cash,
        mtm_pnl=mtm,
    )


def print_report(result: BacktestResult) -> None:
    if result.elapsed_sec > 0:
        tps = result.ticks_processed / result.elapsed_sec
    else:
        tps = 0.0

    print("\n[backtest] done")
    print(f"  ticks processed : {result.ticks_processed}")
    print(f"  elapsed time    : {result.elapsed_sec:.4f} s")
    print(f"  ticks / second  : {int(tps):,} / s\n")
    print(f"  final position  : {result.final_position:.2f}")
    print(f"  last price      : {result.last_price:.2f}")
    print(f"  cash PnL        : {result.cash_pnl:.2f}")
    print(f"  MTM PnL         : {result.mtm_pnl:.2f}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nano-L1 Python backtester (file or synthetic ticks)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to tick CSV (ts,symbol,price,side,qty)",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="If >0, generate this many synthetic ticks instead of reading a file",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="TEST",
        help="Symbol for synthetic ticks (default: TEST)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Optional cap on number of ticks to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic mode (default: 42)",
    )

    args = parser.parse_args()

    # Decide mode: synthetic vs file
    use_synth = args.synthetic and args.synthetic > 0

    if use_synth:
        # Synthetic mode
        n = args.synthetic
        random.seed(args.seed)
        print(
            f"[backtest] synthetic mode: n={n}, symbol={args.symbol}, seed={args.seed}"
        )
        ticks_iter = iter_synthetic_ticks(n=n, symbol=args.symbol)
        result = run_strategy(ticks_iter, max_ticks=args.max_ticks)
        print_report(result)
        return

    # File mode (backwards-compatible with what you ran before)
    if not args.file:
        raise SystemExit("Either --synthetic N or --file PATH must be provided")

    csv_path = args.file
    print(f"[backtest] loading from: {csv_path}")

    ticks_iter = iter_csv_ticks(csv_path, max_ticks=args.max_ticks)
    result = run_strategy(ticks_iter, max_ticks=args.max_ticks)
    print_report(result)


if __name__ == "__main__":
    main()
