import random

class RandomTaker:
    """
    Simple MVP strategy:
    - with probability p, send a market order in tick direction
    """
    def __init__(self, p=0.2, qty=1.0):
        self.p = p
        self.qty = qty
        self._id = 0

    def next_order(self, tick_row):
        if random.random() > self.p:
            return None

        self._id += 1
        side = tick_row["side"]  # buy or sell from tick
        return {
            "id": f"rt_{self._id}",
            "ts": int(tick_row["ts"]),
            "symbol": tick_row["symbol"],
            "side": side,
            "type": "market",
            "qty": float(self.qty),
            # price ignored for market
        }
