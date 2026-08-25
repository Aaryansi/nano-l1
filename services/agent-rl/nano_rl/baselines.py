"""baseline policies for the binary-market mdp.

the spec requires four: random, buy-and-hold, the existing mean-reversion rule,
and the existing LogisticRegression model. a fifth, always-flat, is added
because it is the analytically optimal policy whenever there is no exploitable
signal, and phase 2 measured the real corpus as having none. without it the
comparison table would have no correct answer in it.

the mean-reversion and logistic-regression baselines are ports of
services/agent-py, which is left untouched as the spec requires. porting is
necessary rather than optional: the originals consume a rolling list of trade
prices and emit BUY/SELL/HOLD, while this env consumes an observation vector
and expects a target position. the port preserves the decision rule and its
parameters; what changes is the plumbing.

every policy implements `act(obs, rng) -> int` so the evaluation harness can
treat them uniformly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from nano_rl.env.binary_market import ACTION_TO_TARGET
from nano_rl.env.features import MARKET_FEATURES, feature_index

SHORT, FLAT, LONG = 0, 1, 2


class Policy(Protocol):
    """common interface. `rng` is passed explicitly so every stochastic policy
    is reproducible from the harness's seed rather than from global state."""

    name: str

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int: ...

    def reset(self) -> None: ...


class AlwaysFlat:
    """never trade. optimal whenever there is no exploitable edge.

    this is the baseline that matters most on the real corpus: it earns exactly
    zero by construction, pays no fees, and any strategy that fails to beat it
    is destroying value.
    """

    name = "always-flat"

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        return FLAT

    def reset(self) -> None:
        pass


class BuyAndHold:
    """take a long position at the first decision and hold to settlement.

    the natural "market exposure" baseline. on a binary contract this is a bet
    that the event resolves yes, priced at whatever the market asks.
    """

    name = "buy-and-hold"

    def __init__(self) -> None:
        self._opened = False

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        self._opened = True
        return LONG

    def reset(self) -> None:
        self._opened = False


class RandomPolicy:
    """uniform over the action set. the churn baseline.

    included to show what frictions cost a policy with no information: phase 2
    measured it at -15.13 per episode with costs against +3.53 without.
    """

    name = "random"

    def __init__(self, n_actions: int = len(ACTION_TO_TARGET)) -> None:
        self.n_actions = n_actions

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n_actions))

    def reset(self) -> None:
        pass


class MeanReversion:
    """port of MeanReversionPolicy from services/agent-py/agent.py.

    the original keeps a rolling window of trade prices and compares the last
    price to the window mean, in basis points:

        price below mean by more than threshold -> BUY
        price above mean by more than threshold -> SELL
        otherwise                               -> HOLD, with epsilon-probability
                                                   exploration that alternates side

    the port preserves that rule, the 5 bp threshold, and the alternating
    exploration. the window is reconstructed from the observation rather than
    from a price list: `implied_prob` is the current mid and `p_change_2` is its
    backward two-step difference, so mid minus p_change_2 is the mid two steps
    ago, which stands in for the window mean over the same horizon.

    this is a faithful port of the decision rule, not of the original's exact
    20-tick window, because the 60s decision grid only affords 14 steps per
    episode. the deviation is recorded here rather than hidden.
    """

    name = "mean-reversion"

    def __init__(self, threshold_bp: float = 5.0, epsilon: float = 0.25) -> None:
        self.threshold_bp = threshold_bp
        self.epsilon = epsilon
        self._flip = 1
        self._i_mid = feature_index("implied_prob")
        self._i_change2 = feature_index("p_change_2")

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        mid = float(obs[self._i_mid])
        change2 = float(obs[self._i_change2])
        reference = mid - change2  # the mid two steps ago

        if reference <= 0:
            return FLAT

        diff_bp = (mid - reference) / reference * 10_000.0

        # price below its recent level: expect reversion upward, so go long
        if diff_bp <= -self.threshold_bp:
            return LONG
        if diff_bp >= self.threshold_bp:
            return SHORT

        # flat region: the original explores here, alternating side
        if rng.random() < self.epsilon:
            self._flip *= -1
            return LONG if self._flip > 0 else SHORT
        return FLAT

    def reset(self) -> None:
        self._flip = 1


class LogisticBaseline:
    """port of BinanceMLPolicy from services/agent-py/agent.py.

    the original loads a pickled sklearn LogisticRegression trained on binance
    trades with the feature vector [rel_move, last_price, volatility,
    window_len], calls predict_proba, and acts when the winning class clears a
    0.45 confidence threshold, otherwise falling back to a momentum rule.

    two honest notes about this baseline.

    first, the shipped model was trained on BTC spot prices in the tens of
    thousands, and `last_price` is one of its four features. feeding it a
    binary contract price in [0, 1] is far outside its training distribution.
    that is a property of the original model, not of this port, and it is
    exactly why the port also supports refitting on this corpus.

    second, when no model is supplied this falls back to the same momentum rule
    the original uses when unsure, so the baseline is always available.
    """

    name = "logistic"

    def __init__(
        self,
        model=None,
        prob_threshold: float = 0.45,
        window_len: float = 50.0,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.prob_threshold = prob_threshold
        # the shipped model was trained with window=50 and its LARGEST
        # coefficient by two orders of magnitude is on this feature, which was
        # a constant throughout its training set. passing this env's 14 instead
        # of the trained 50 flips its output to 94% BUY on any input. we pass
        # 50 so the baseline is evaluated at the operating point it was fit at,
        # which is the charitable reading.
        self.window_len = window_len
        if name:
            self.name = name
        self._i_mid = feature_index("implied_prob")
        self._i_change1 = feature_index("p_change_1")
        self._i_vol = feature_index("p_realized_vol")

    def _features(self, obs: np.ndarray) -> np.ndarray:
        mid = float(obs[self._i_mid])
        change = float(obs[self._i_change1])
        reference = mid - change
        rel_move = (mid - reference) / reference if reference != 0 else 0.0
        return np.array([[rel_move, mid, float(obs[self._i_vol]), self.window_len]])

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        if self.model is not None and hasattr(self.model, "predict_proba"):
            x = self._features(obs)
            proba = self.model.predict_proba(x)[0]
            classes = list(self.model.classes_)

            def p(label: int) -> float:
                return float(proba[classes.index(label)]) if label in classes else 0.0

            p_sell, p_hold, p_buy = p(-1), p(0), p(1)
            if p_buy > self.prob_threshold and p_buy > p_sell and p_buy > p_hold:
                return LONG
            if p_sell > self.prob_threshold and p_sell > p_buy and p_sell > p_hold:
                return SHORT

        # momentum fallback, matching the original's behaviour when unsure
        rel_move = float(self._features(obs)[0, 0])
        if rel_move > 0.0002:
            return SHORT
        if rel_move < -0.0002:
            return LONG
        return FLAT

    def reset(self) -> None:
        pass


def fit_logistic_on_corpus(batch, horizon: int = 3, threshold: float = 0.0005):
    """refit the agent-py logistic method on THIS corpus's training split.

    the shipped pickle is a poor baseline for two independent reasons, so a
    refit is the fair comparison rather than a courtesy:

      it was trained on BTC spot prices in the tens of thousands, and
      `last_price` is one of its four features. binary contract prices live in
      [0, 1], entirely outside that range.

      its dominant coefficient is on `window_len`, which was constant across
      its entire training set. a coefficient on a constant is a bias term, so
      the model is close to a constant classifier that merely shifts when the
      window changes.

    the labelling scheme is the original's: sign of the mid change over
    `horizon` steps, with a dead zone. using future labels to FIT on the train
    split is ordinary supervised learning, not lookahead; the resulting policy
    is still evaluated causally, one observation at a time.

    returns None if sklearn is unavailable or the labels are degenerate.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    feats = batch.market_features()
    i_mid = MARKET_FEATURES.index("implied_prob")
    i_ch = MARKET_FEATURES.index("p_change_1")
    i_vol = MARKET_FEATURES.index("p_realized_vol")

    x_rows, y_rows = [], []
    n_steps = feats.shape[1]
    for ep in range(feats.shape[0]):
        for t in range(n_steps - horizon):
            mid = float(feats[ep, t, i_mid])
            ref = mid - float(feats[ep, t, i_ch])
            rel = (mid - ref) / ref if ref != 0 else 0.0
            future = float(feats[ep, t + horizon, i_mid])
            change = (future - mid) / mid if mid != 0 else 0.0
            label = 1 if change > threshold else (-1 if change < -threshold else 0)
            x_rows.append([rel, mid, float(feats[ep, t, i_vol]), 50.0])
            y_rows.append(label)

    x = np.asarray(x_rows)
    y = np.asarray(y_rows)
    if len(set(y.tolist())) < 2:
        return None

    clf = LogisticRegression(max_iter=500)  # multi_class removed in sklearn 1.8
    clf.fit(x, y)
    return clf


def default_baselines(logistic_model=None, logistic_refit=None) -> list[Policy]:
    """the full comparison set, in the order the report presents them."""
    out: list[Policy] = [
        AlwaysFlat(),
        BuyAndHold(),
        RandomPolicy(),
        MeanReversion(),
        LogisticBaseline(model=logistic_model, name="logistic-shipped"),
    ]
    if logistic_refit is not None:
        out.append(LogisticBaseline(model=logistic_refit, name="logistic-refit"))
    return out
