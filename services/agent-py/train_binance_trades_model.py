#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def build_dataset(df: pd.DataFrame, window: int, horizon: int):
    """
    Turn a price series into supervised features/labels.

    Features (per window):
      - rel_move: (last - mean) / mean
      - last: last price
      - volatility: max(window) - min(window)
      - length: window size (constant)

    Labels:
      - +1 if price horizon steps ahead is up > +0.05%
      - -1 if down < -0.05%
      -  0 otherwise
    """
    prices = df["price"].astype(float).values
    X, y = [], []

    for i in range(window, len(prices) - horizon):
        w = prices[i - window : i]
        last = w[-1]
        mean = w.mean() if window > 0 else last
        rel_move = (last - mean) / mean if mean != 0 else 0.0
        volatility = w.max() - w.min()
        length = window

        future_price = prices[i + horizon - 1]
        change = (future_price - last) / last if last != 0 else 0.0

        if change > 0.0005:
            label = 1
        elif change < -0.0005:
            label = -1
        else:
            label = 0

        X.append([rel_move, last, volatility, length])
        y.append(label)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Train Binance BTCUSDT trade model")
    parser.add_argument("--csv", required=True, help="Path to agg trades CSV")
    parser.add_argument("--window", type=int, default=50, help="window size")
    parser.add_argument("--horizon", type=int, default=10, help="prediction horizon")
    parser.add_argument("--out", required=True, help="output model path (.pkl)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "price" not in df.columns:
        raise SystemExit("CSV missing 'price' column")

    print(f"[train] loaded {len(df)} trades from {csv_path}")

    X, y = build_dataset(df, args.window, args.horizon)
    print(f"[train] dataset shapes: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        max_iter=200,
        # keep it simple/compatible
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("[train] classification report:\n")
    print(classification_report(y_test, y_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:        # IMPORTANT: binary mode
        pickle.dump(clf, f)

    print(f"[train] saved model to {out_path.resolve()}")


if __name__ == "__main__":
    main()
