"""causal feature construction for the binary-market mdp.

features are split into two groups because they have different provenance and
different failure modes:

  market features   derived only from the episode arrays, which are already
                    causal by construction (see nano_rl/data/episode.py).

  position features derived from the agent's own inventory, which is causal
                    trivially, but which must be recomputed every step rather
                    than precomputed, since it depends on the policy.

normalisation statistics are fit on the training split only and then frozen.
fitting them on the full corpus would leak test-period distribution into
training, which is a subtle and very common form of lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# kalshi-derived features, precomputable per episode.
KALSHI_FEATURES: tuple[str, ...] = (
    "implied_prob",
    "spread",
    "p_change_1",
    "p_change_2",
    "p_realized_vol",
    "flow_imbalance",
    "volume_rate",
    "staleness",
    "time_to_expiry_frac",
)

# binance-spot-derived features. the contract resolves on the sign of the spot
# move, so these describe the underlying being predicted. see
# nano_rl/data/binance.py. when spot coverage is missing for an episode these
# are zero-filled rather than imputed, and the corpus records that fact.
SPOT_FEATURES: tuple[str, ...] = (
    "spot_ret_since_open",
    "spot_ret_30s",
    "spot_ret_60s",
    "spot_realized_vol",
    "spot_implied_gap",
)

# the full precomputable block, which is what the normalizer is fit on.
MARKET_FEATURES: tuple[str, ...] = KALSHI_FEATURES + SPOT_FEATURES

# position features, recomputed each step from agent state. these are already
# on a unit scale by construction and are NOT normalised; see the env's _obs.
POSITION_FEATURES: tuple[str, ...] = (
    "position",
    "avg_entry_price",
    "unrealized_pnl",
    "time_in_position",
)

FEATURE_NAMES: tuple[str, ...] = MARKET_FEATURES + POSITION_FEATURES
N_FEATURES = len(FEATURE_NAMES)
N_KALSHI = len(KALSHI_FEATURES)
N_SPOT = len(SPOT_FEATURES)


def build_market_features(
    bid: np.ndarray,
    ask: np.ndarray,
    last_price: np.ndarray,
    volume: np.ndarray,
    staleness: np.ndarray,
    flow_imbalance: np.ndarray,
    t_sec: np.ndarray,
    duration_s: float,
    vol_window: int = 3,
) -> np.ndarray:
    """compute the per-step market feature block for one episode.

    args:
        bid, ask, last_price, volume, staleness, flow_imbalance: aligned
            per-step arrays from an Episode.
        t_sec: decision-boundary times, seconds since open.
        duration_s: total episode length, for the time-to-expiry fraction.
        vol_window: trailing window, in steps, for realised volatility.

    returns:
        (n_steps, len(MARKET_FEATURES)) float array.

    every lag below uses np.diff-style backward differences padded at the
    front, never a centred or forward difference, which would look ahead.
    """
    n = len(t_sec)
    mid = 0.5 * (bid + ask)

    # backward differences over 1 and 2 steps, front-padded with zero so that
    # step 0 sees "no change" rather than a value borrowed from the future.
    def backward_diff(x: np.ndarray, lag: int) -> np.ndarray:
        out = np.zeros_like(x)
        if n > lag:
            out[lag:] = x[lag:] - x[:-lag]
        return out

    p_change_1 = backward_diff(mid, 1)
    p_change_2 = backward_diff(mid, 2)

    # trailing realised vol: stdev over the previous `vol_window` steps,
    # inclusive of the current step and nothing after it.
    p_vol = np.zeros(n)
    for i in range(n):
        lo = max(0, i - vol_window + 1)
        window = mid[lo : i + 1]
        p_vol[i] = window.std() if len(window) > 1 else 0.0

    # volume is heavy-tailed across six orders of magnitude, so compress it.
    volume_rate = np.log1p(np.maximum(volume, 0.0))

    time_to_expiry = (duration_s - t_sec) / duration_s

    return np.column_stack(
        [
            mid,  # implied_prob
            ask - bid,  # spread
            p_change_1,
            p_change_2,
            p_vol,
            flow_imbalance,
            volume_rate,
            staleness,
            time_to_expiry,
        ]
    )


@dataclass
class PositionState:
    """the agent's inventory, in yes-equivalent contracts.

    a short-yes position is economically a long-no position; we track a single
    signed quantity and let the cost model handle the side mapping at
    execution time.
    """

    position: float = 0.0
    avg_entry_price: float = 0.0
    steps_in_position: int = 0

    def unrealized_pnl(self, mark: float) -> float:
        """mark-to-market pnl of the open position at price `mark`."""
        if self.position == 0.0:
            return 0.0
        return self.position * (mark - self.avg_entry_price)

    def apply_fill(self, delta: float, price: float) -> float:
        """update inventory for a fill, returning realised pnl from any close.

        handles the four cases explicitly rather than with a clever formula,
        because getting the average-entry update wrong on a position flip is a
        classic source of silently wrong pnl.
        """
        if delta == 0.0:
            self.steps_in_position += 1
            return 0.0

        old_pos = self.position
        new_pos = old_pos + delta
        realized = 0.0

        if old_pos == 0.0 or np.sign(delta) == np.sign(old_pos):
            # opening, or adding to an existing position: blend the entry.
            total = abs(old_pos) + abs(delta)
            self.avg_entry_price = (
                abs(old_pos) * self.avg_entry_price + abs(delta) * price
            ) / total
        elif abs(delta) < abs(old_pos):
            # partial close: realise on the closed portion, entry unchanged.
            realized = abs(delta) * (price - self.avg_entry_price) * np.sign(old_pos)
        else:
            # full close, possibly flipping through zero.
            realized = abs(old_pos) * (price - self.avg_entry_price) * np.sign(old_pos)
            # whatever remains beyond the close opens a fresh position.
            self.avg_entry_price = price if new_pos != 0.0 else 0.0

        self.position = new_pos
        if new_pos == 0.0:
            self.avg_entry_price = 0.0
            self.steps_in_position = 0
        else:
            self.steps_in_position = 0 if np.sign(new_pos) != np.sign(old_pos) else self.steps_in_position + 1

        return float(realized)

    def features(self, mark: float, max_position: float, max_steps: int) -> np.ndarray:
        """the position feature block, normalised to roughly unit scale."""
        return np.array(
            [
                self.position / max_position,
                self.avg_entry_price,
                self.unrealized_pnl(mark) / max_position,
                min(self.steps_in_position / max(max_steps, 1), 1.0),
            ]
        )


class FeatureNormalizer:
    """z-score normaliser fit on the training split only.

    deliberately not a sklearn StandardScaler: keeping it here makes the
    train-only fit explicit and keeps the env dependency-light.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "FeatureNormalizer":
        """fit on a (n_samples, n_features) block from the TRAIN split only."""
        self.mean = x.mean(axis=0)
        # guard against zero-variance columns, which would produce inf.
        self.std = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("normalizer used before fit; this would leak or crash")
        return (x - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        if self.mean is None or self.std is None:
            raise RuntimeError("nothing to serialise")
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, d: dict[str, list[float]]) -> "FeatureNormalizer":
        self.mean = np.asarray(d["mean"])
        self.std = np.asarray(d["std"])
        return self
