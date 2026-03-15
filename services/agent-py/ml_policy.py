from typing import List, Literal
import joblib
import numpy as np

Action = Literal["BUY", "SELL", "HOLD"]


class MLPolicy:
    """sklearn model wrapper using [last, mean, std, z_score] features"""

    def __init__(self, model_path: str, window: int = 50):
        self.model = joblib.load(model_path)
        self.window = window

    def act(self, prices: List[float]) -> Action:
        if len(prices) < self.window:
            # not enough history -> do nothing
            return "HOLD"

        window_slice = np.array(prices[-self.window :], dtype=float)
        last = float(window_slice[-1])
        mean = float(window_slice.mean())
        std = float(window_slice.std() + 1e-8)
        z = (last - mean) / std

        X = np.array([[last, mean, std, z]], dtype=float)
        pred = int(self.model.predict(X)[0])

        if pred > 0:
            return "BUY"
        elif pred < 0:
            return "SELL"
        else:
            return "HOLD"
