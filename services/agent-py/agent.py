#!/usr/bin/env python
import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional

import pickle
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import requests

from ml_policy import MLPolicy

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass
class Trade:
    id: int
    symbol: str
    price: float
    qty: float
    aggressor_side: str  # "buy" or "sell"
    ts: int


class MeanReversionPolicy:
    """
    Simple placeholder policy with an RL-style interface.

    You can later replace this with a trained RL model that implements:
        act(self, state: List[float]) -> Action
    """

    def __init__(
        self,
        window: int = 20,
        threshold_bp: float = 5.0,
        epsilon: float = 0.25,  # 25% random exploration so we SEE trades
    ):
        """
        window        : how many prices to use for the moving average
        threshold_bp  : how many basis points (0.01%) away from mean to trigger
                        a trade, e.g. 5 bp = 0.05%
        epsilon       : probability of taking a random BUY/SELL when flat
        """
        import random

        self.window = window
        self.threshold_bp = threshold_bp
        self.epsilon = epsilon
        self._flip = 1  # just to alternate when exploring
        self._rng = random.Random(42)

    def act(self, prices: List[float]) -> Action:
        if not prices:
            return "HOLD"

        last = prices[-1]

        # If we don't have enough history yet, just occasionally explore
        if len(prices) < self.window:
            if self._rng.random() < self.epsilon:
                self._flip *= -1
                return "BUY" if self._flip > 0 else "SELL"
            return "HOLD"

        mean = sum(prices[-self.window :]) / self.window
        if mean == 0:
            return "HOLD"

        diff_bp = (last - mean) / mean * 10_000.0

        # Price below mean -> go long (BUY)
        if diff_bp <= -self.threshold_bp:
            return "BUY"

        # Price above mean -> go short (SELL)
        if diff_bp >= self.threshold_bp:
            return "SELL"

        # --- exploration when basically flat ---
        if self._rng.random() < self.epsilon:
            self._flip *= -1
            return "BUY" if self._flip > 0 else "SELL"

        return "HOLD"
    

class BinanceMLPolicy:
    """
    Policy that uses a trained sklearn model on a rolling price window.

    Model expects features:
      [rel_move, last, volatility, length]
    where:
      rel_move = (last - mean) / mean
      volatility = max(window) - min(window)
      length = window size (constant)
    """

    def __init__(self, model, window: int):
        self.model = model
        self.window = window

    def decide(self, prices: Deque[float]) -> str:
        if len(prices) < self.window:
            return "HOLD"

        w = np.array(list(prices)[-self.window:])
        last = w[-1]
        mean = w.mean() if len(w) > 0 else last
        rel_move = (last - mean) / mean if mean != 0 else 0.0
        volatility = w.max() - w.min()
        length = self.window

        X = np.array([[rel_move, last, volatility, length]], dtype=float)
        pred = self.model.predict(X)[0]

        if pred == 1:
            return "BUY"
        elif pred == -1:
            return "SELL"
        return "HOLD"


class EngineClient:
    def __init__(self, base_url: str, symbol: str, order_qty: float):
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self.order_qty = order_qty

        # IMPORTANT: must match your dashboard OrderPanel
        # which posts to `${API_BASE}/order`.
        self.order_endpoint = f"{self.base_url}/order"

    def fetch_last_trade(self) -> Optional[Trade]:
        url = f"{self.base_url}/api/trades/recent?limit=1"
        resp = requests.get(url, timeout=2.0)
        resp.raise_for_status()
        payload = resp.json()

        trades = payload.get("trades") if isinstance(payload, dict) else payload
        if not trades:
            return None

        t = trades[0]
        return Trade(
            id=t["id"],
            symbol=t["symbol"],
            price=float(t["price"]),
            qty=float(t.get("qty", 0)),
            aggressor_side=str(
                t.get("aggressor_side") or t.get("aggressorSide") or ""
            ),
            ts=int(t.get("ts", 0)),
        )

    def submit_order(self, side: Action, price: float) -> dict:
        """
        Sends a *simple* limit order identical in shape to what your front-end
        sends. Adjust fields to match your engine's API if needed.
        """
        body = {
            "id": f"agent-{int(time.time() * 1000)}",
            "symbol": self.symbol,
            "side": side.lower(),  # "buy" / "sell"
            "type": "limit",
            "price": price,
            "qty": self.order_qty,
        }
        resp = requests.post(self.order_endpoint, json=body, timeout=2.0)
        resp.raise_for_status()
        return resp.json()


@dataclass
class AgentConfig:
    engine_url: str
    symbol: str
    qty: float
    poll_interval: float
    history_len: int
    model_path: Optional[str] = None

class TradingAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.client = EngineClient(cfg.engine_url, cfg.symbol, cfg.qty)
        self.prices: Deque[float] = deque(maxlen=cfg.history_len)

        # --- choose policy ---
        if cfg.model_path:
            try:
                with open(cfg.model_path, "rb") as f:
                    model = pickle.load(f)
                # use Binance ML policy
                self.policy = BinanceMLPolicy(model=model, window=min(50, cfg.history_len))
                print(f"[agent] using BinanceMLPolicy with model={cfg.model_path}")
            except Exception as e:
                print(f"[agent] WARNING: failed to load model at {cfg.model_path}: {e}")
                print("[agent] falling back to MeanReversionPolicy")
                self.policy = MeanReversionPolicy(
                    window=min(50, cfg.history_len),
                    threshold_bp=5.0,
                )
        else:
            self.policy = MeanReversionPolicy(
                window=min(50, cfg.history_len),
                threshold_bp=5.0,
            )
            print("[agent] using fallback MeanReversionPolicy")

        self.position = 0.0
        self.cash = 0.0
        self.last_price = None


    def step(self):
        # 1) Pull latest trade from engine
        trade = self.client.fetch_last_trade()
        if not trade:
            print("[agent] no trades yet, skipping")
            return

        self.last_price = trade.price
        self.prices.append(trade.price)

        prices_list = list(self.prices)
        action: Action = self.policy.act(prices_list)

        mean_price = sum(prices_list) / len(prices_list)
        print(
            f"[agent] last_price={trade.price:.4f}, "
            f"mean({len(prices_list)})={mean_price:.4f}, "
            f"action={action}"
        )

        # 2) Send order if BUY/SELL
        if action in ("BUY", "SELL"):
            try:
                print(
                    f"[agent] placing {action} {self.cfg.qty} @ {trade.price:.4f}"
                )
                resp = self.client.submit_order(action, trade.price)
                print("[agent] order ok:", resp)
            except Exception as exc:
                print("[agent] order error:", exc)

        # 3) Very rough local PnL accounting assuming we got filled at trade.price.
        if action == "BUY":
            self.position += self.cfg.qty
            self.cash -= self.cfg.qty * trade.price
        elif action == "SELL":
            self.position -= self.cfg.qty
            self.cash += self.cfg.qty * trade.price

        # Mark-to-market
        mtm = self.cash + (self.position * trade.price)
        print(
            f"[agent] position={self.position:.4f}, cash={self.cash:.2f}, mtm={mtm:.2f}"
        )

    def run_forever(self):
        print(
            f"[agent] starting for symbol={self.cfg.symbol}, "
            f"engine={self.cfg.engine_url}, qty={self.cfg.qty}, "
            f"policy={type(self.policy).__name__}"
        )
        while True:
            try:
                self.step()
            except Exception as exc:
                print("[agent] step error:", exc)
            time.sleep(self.cfg.poll_interval)


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description="Nano-L1 ML/mean-reversion agent")
    parser.add_argument(
        "--engine-url",
        default=os.getenv("ENGINE_HTTP_URL", "http://localhost:8080"),
        help="Base URL for Go engine HTTP (default: %(default)s)",
    )
    parser.add_argument(
        "--symbol",
        default="TEST",
        help="Symbol to trade (default: %(default)s)",
    )
    parser.add_argument(
        "--qty",
        type=float,
        default=1.0,
        help="Order size (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between decisions (default: %(default)s)",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=200,
        help="How many prices to keep in state (default: %(default)s)",
    )
    parser.add_argument(
        "--policy",
        choices=["mean_rev", "ml"],
        default="mean_rev",
        help="Which policy to use: mean_rev or ml (default: mean_rev)",
    )

    parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to pickled sklearn model (optional)",
)
    
    

    args = parser.parse_args()
    return AgentConfig(
    engine_url=args.engine_url,
    symbol=args.symbol,
    qty=args.qty,
    poll_interval=args.poll_interval,
    history_len=args.history_len,
    model_path=args.model_path,
)

class BinanceMLPolicy:
    """
    Policy that uses a trained sklearn model on a rolling price window.

    Model expects features: [rel_move, last, volatility, length]
      rel_move   = (last - mean) / mean
      volatility = max(window) - min(window)
      length     = window size (constant)
    """

    def __init__(self, model, window: int = 50, prob_threshold: float = 0.45):
        self.model = model
        self.window = window
        self.prob_threshold = prob_threshold

    def act(self, last_price: float, mean_price: float, prices: Deque[float]) -> str:
        """
        This matches how TradingAgent calls policy.act(...).
        last_price / mean_price are passed in but we also
        use the full `prices` deque to build features.
        """
        # Not enough history yet → do nothing
        if len(prices) < self.window:
            return "HOLD"

        # Take last `window` prices
        w = np.array(list(prices)[-self.window:])
        last = w[-1]
        mean = w.mean() if w.size > 0 else last

        # Features
        rel_move = (last - mean) / mean if mean != 0 else 0.0
        volatility = float(w.max() - w.min())
        length = float(self.window)

        X = np.array([[rel_move, last, volatility, length]], dtype=float)

        # Probabilities for each class
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)  # e.g. [-1, 0, 1]

        def p_for(label):
            return float(proba[classes.index(label)]) if label in classes else 0.0

        p_sell = p_for(-1)
        p_hold = p_for(0)
        p_buy = p_for(1)

        # Confident ML decisions
        if p_buy > self.prob_threshold and p_buy > p_sell and p_buy > p_hold:
            return "BUY"
        if p_sell > self.prob_threshold and p_sell > p_buy and p_sell > p_hold:
            return "SELL"

        # Fallback: tiny momentum rule so it actually trades sometimes
        if rel_move > 0.0002:
            return "SELL"
        if rel_move < -0.0002:
            return "BUY"

        return "HOLD"



def main():
    cfg = parse_args()
    agent = TradingAgent(cfg)
    agent.run_forever()


if __name__ == "__main__":
    main()
