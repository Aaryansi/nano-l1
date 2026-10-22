"""Empirical-row and nearest-neighbor input-replacement interventions.

Marginal replacement takes missing coordinates from an empirical background
row. Conditional replacement selects donors among nearest neighbors in retained
coordinates. This is an approximation, not a known conditional feature law.

For full replacement, the input row comes from empirical background support.
That fact alone does not ensure compatibility with the recipient's inventory,
reward, history or subsequent observations. Repeated row draws can also break
sequence structure. Neither endpoint support nor nearest-neighbor matching
establishes immunity to distribution shift in an active decision process.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch

from nano_rl.agents.ppo import PPOAgent
from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.features import FeatureNormalizer
from nano_rl.explain.rollout import VectorizedRollout

Mode = Literal["marginal", "conditional"]


def _scales(background: np.ndarray) -> np.ndarray:
    """per-feature spread, used to make the neighbour distance scale free.

    without this the distance is dominated by whichever feature happens to have
    the widest range, and the "nearest" rows are near in that feature alone.
    """
    s = background.std(axis=0)
    return np.where(s < 1e-8, 1.0, s)


def draw_replacements(
    obs: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    mode: Mode,
    k: int,
    scales: np.ndarray,
) -> np.ndarray:
    """pick one background row per observation and return the full rows.

    the caller substitutes only the masked columns. returning whole rows keeps
    the masked block internally consistent: it comes from one real state rather
    than being assembled feature-by-feature from different ones.
    """
    n = len(obs)
    if mode == "marginal" or not mask.any() or mask.all():
        # nothing to condition on (or nothing to replace): the modes coincide
        return background[rng.integers(0, len(background), size=n)]

    # distance in the kept features only, standardised
    kept = np.flatnonzero(mask)
    a = obs[:, kept] / scales[kept]
    b = background[:, kept] / scales[kept]
    d2 = (
        (a * a).sum(1)[:, None]
        - 2.0 * a @ b.T
        + (b * b).sum(1)[None, :]
    )
    kk = int(min(k, len(background)))
    near = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    pick = near[np.arange(n), rng.integers(0, kk, size=n)]
    return background[pick]


def masked_value_fn(
    agent: PPOAgent,
    batch: EpisodeBatch,
    background: np.ndarray,
    normalizer: FeatureNormalizer | None,
    max_position: float,
    n_episodes: int,
    seed: int,
    mode: Mode = "marginal",
    k: int = 16,
) -> Callable[[np.ndarray], float]:
    """v(S) under the chosen masking mode.

    deliberately mirrors outcome_schemes._masked_value_fn so that mode is the
    only thing that differs between the two arms of the comparison.
    """
    sub = batch.subset(np.arange(min(n_episodes, len(batch))))
    roll = VectorizedRollout(sub, normalizer=normalizer, max_position=max_position)
    scales = _scales(background)

    def v(mask: np.ndarray) -> float:
        rng = np.random.default_rng(seed)

        def policy(obs: np.ndarray) -> np.ndarray:
            synthetic = obs.copy()
            if not mask.all():
                draws = draw_replacements(
                    obs, mask, background, rng, mode, k, scales
                )
                synthetic[:, ~mask] = draws[:, ~mask]
            with torch.no_grad():
                logits, _ = agent.net(torch.as_tensor(synthetic, dtype=torch.float32))
                return logits.argmax(dim=-1).numpy()

        return float(roll.run(policy)["returns"].mean())

    return v


def offmanifold_distance(
    obs: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    mode: Mode,
    k: int = 16,
) -> float:
    """mean standardised distance from a synthetic state to its nearest real one.

    this is the quantity the objection is really about. a synthetic state that
    sits on top of a real one is a state the policy could have met in training;
    one that sits far from every real state is not.

    measured against `background` as the sample of real states, in units of
    per-feature standard deviations, averaged over features so the number does
    not grow with dimension.
    """
    scales = _scales(background)
    draws = draw_replacements(obs, mask, background, rng, mode, k, scales)
    synthetic = obs.copy()
    synthetic[:, ~mask] = draws[:, ~mask]

    a = synthetic / scales
    b = background / scales
    d2 = (
        (a * a).sum(1)[:, None]
        - 2.0 * a @ b.T
        + (b * b).sum(1)[None, :]
    )
    nearest = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return float(nearest.mean() / np.sqrt(obs.shape[1]))
