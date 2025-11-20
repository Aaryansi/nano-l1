import os
import time
import requests
from nano_backtest.data.loaders import load_ticks_csv
from nano_backtest.strategies.random_taker import RandomTaker
from nano_backtest.metrics.pnl import update_pnl, mark_to_market

ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8080/order")
DATA_PATH  = os.getenv("DATA_PATH", "../../data/sample/sample_ticks.csv")

def send_order(order):
    r = requests.post(ENGINE_URL, json=order, timeout=5)
    r.raise_for_status()
    return r.json()

def main():
    print("ENGINE_URL:", ENGINE_URL)
    print("DATA_PATH :", DATA_PATH)

    df = load_ticks_csv(DATA_PATH)
    strat = RandomTaker(p=0.5, qty=1.0)

    pos, cash = 0.0, 0.0
    equity_curve = []
    last_price = None

    t0 = time.time()

    for _, row in df.iterrows():
        last_price = float(row["price"])

        order = strat.next_order(row)
        if order is None:
            continue

        resp = send_order(order)
        trades = resp.get("trades") or []
        pos, cash = update_pnl(pos, cash, trades)

        eq = mark_to_market(pos, cash, last_price)
        equity_curve.append(eq)

    dt = time.time() - t0
    final_eq = equity_curve[-1] if equity_curve else 0.0

    print(f"Backtest done in {dt:.3f}s")
    print(f"Final Position: {pos:.2f}")
    print(f"Final Cash: {cash:.2f}")
    print(f"Final Equity (MtM): {final_eq:.2f}")
    print(f"Num Orders Sent: {len(equity_curve)}")

if __name__ == "__main__":
    main()
