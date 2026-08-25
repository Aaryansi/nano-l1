"""walk-forward train/val/test splits.

the split is the second place (after nano_rl/data/episode.py) where lookahead
creeps in, and it does so in ways that survive code review:

  - shuffling. a random split on a time series puts adjacent, near-identical
    samples in both train and test. reported accuracy then measures
    memorisation, not generalisation.

  - boundary bleed. features with trailing windows computed near a split
    boundary can incorporate rows from the neighbouring split. a purge gap
    removes the affected episodes rather than hoping the window is short.

  - normalisation fit on the pooled corpus, which leaks test-period
    distribution into training. the normalizer is fit here, on train only.

this module therefore returns splits AND the fitted normalizer together, so
that callers cannot accidentally fit on the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.features import FeatureNormalizer


@dataclass(frozen=True)
class WalkForwardSplit:
    """contiguous, time-ordered, disjoint episode splits."""

    train: EpisodeBatch
    val: EpisodeBatch
    test: EpisodeBatch
    normalizer: FeatureNormalizer

    # kept for reporting: how many episodes the purge gaps removed
    n_purged: int

    def summary(self) -> str:
        def span(b: EpisodeBatch) -> str:
            lo = np.min(b.open_epoch)
            hi = np.max(b.open_epoch)
            return f"{len(b):>5} eps, {(hi - lo) / 86400:5.1f}d, yes={b.settlement.mean():.3f}"

        return (
            f"  train: {span(self.train)}\n"
            f"  val  : {span(self.val)}\n"
            f"  test : {span(self.test)}\n"
            f"  purged: {self.n_purged} episodes at boundaries"
        )


def walk_forward_split(
    batch: EpisodeBatch,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    purge: int = 1,
    fit_normalizer: bool = True,
) -> WalkForwardSplit:
    """split a corpus by time into train / val / test.

    args:
        batch: the full corpus.
        train_frac, val_frac: fractions of episodes. test takes the remainder.
        purge: episodes to drop on each side of a boundary. one episode is
            enough here because every trailing feature window is bounded by
            the episode itself (features never cross an episode boundary), but
            the parameter is exposed so the assumption can be stress-tested.
        fit_normalizer: fit z-score stats on the TRAIN split only.

    returns:
        a WalkForwardSplit. the normalizer is unfitted if fit_normalizer is
        False, in which case using it raises rather than silently passing
        through.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must each be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test split")

    n = len(batch)
    if n < 10:
        raise ValueError(f"corpus too small to split: {n} episodes")

    # sort by market open time. the corpus is usually already ordered, but
    # relying on that would be a silent correctness dependency.
    order = np.argsort(batch.open_epoch, kind="stable")

    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    train_idx = order[: max(i_train - purge, 1)]
    val_idx = order[i_train + purge : max(i_val - purge, i_train + purge + 1)]
    test_idx = order[i_val + purge :]

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"split produced an empty partition (n={n}, purge={purge}); "
            "reduce purge or supply more episodes"
        )

    train = batch.subset(train_idx)
    val = batch.subset(val_idx)
    test = batch.subset(test_idx)

    normalizer = FeatureNormalizer()
    if fit_normalizer:
        # flatten (n_episodes, n_steps, n_market_features) to fit over rows.
        # position features are appended at env runtime and are already
        # normalised by construction, so they are not fit here; the env
        # concatenates them post-transform.
        feats = train.market_features()
        normalizer.fit(feats.reshape(-1, feats.shape[-1]))

    n_purged = n - (len(train_idx) + len(val_idx) + len(test_idx))

    return WalkForwardSplit(
        train=train, val=val, test=test, normalizer=normalizer, n_purged=n_purged
    )
