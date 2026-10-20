"""Observation replacements with explicit state and sequence preservation."""
from __future__ import annotations

import numpy as np

from nano_rl.env.features import MARKET_FEATURES, feature_index

MODES = ("full_observation", "market_rows", "market_trajectories")


def intervention_rollout(policy, rollout, background, mode, seed=0):
    """Replay with a fixed intervention, retaining recipient state when named.

    Market modes retain the real clock and all four recipient inventory inputs.
    Rows draw a new same-step donor each step. Trajectories keep one donor per
    recipient throughout the episode. Donor market history is then coherent,
    but its joint consistency with recipient inventory is not guaranteed.
    Retained entry price and unrealized P&L can themselves reveal market data.
    """
    if mode not in MODES:
        raise ValueError(f"unknown observation intervention: {mode}")
    rng = np.random.default_rng(seed)
    donors = (rng.integers(rollout.n_episodes, size=rollout.n_episodes)
              if mode == "market_trajectories" else None)
    time_index = feature_index("time_to_expiry_frac")
    market_columns = np.array([i for i in range(len(MARKET_FEATURES)) if i != time_index])
    t = 0

    def replaced(obs):
        nonlocal t
        x = obs.copy()
        if mode == "full_observation":
            x[:] = background[rng.integers(len(background), size=len(obs))]
        else:
            rows = donors if donors is not None else rng.integers(rollout.n_episodes, size=len(obs))
            market = rollout.market[rows, t, :]
            if rollout.normalizer is not None:
                market = rollout.normalizer.transform(market)
            x[:, market_columns] = market[:, market_columns]
        t += 1
        return policy(x)

    return rollout.run(replaced)
