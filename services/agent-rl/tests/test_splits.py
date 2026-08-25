"""walk-forward split tests, and the normalizer wiring the env depends on."""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.features import MARKET_FEATURES, N_FEATURES

from .conftest import make_batch


@pytest.fixture
def corpus():
    return make_batch(n_episodes=200, n_steps=8)


class TestOrdering:
    def test_splits_are_chronological_and_disjoint(self, corpus) -> None:
        s = walk_forward_split(corpus)
        assert s.train.open_epoch.max() < s.val.open_epoch.min()
        assert s.val.open_epoch.max() < s.test.open_epoch.min()

    def test_no_episode_appears_twice(self, corpus) -> None:
        s = walk_forward_split(corpus)
        allep = np.concatenate(
            [s.train.open_epoch, s.val.open_epoch, s.test.open_epoch]
        )
        assert len(allep) == len(set(allep.tolist()))

    def test_purge_actually_removes_episodes(self, corpus) -> None:
        none = walk_forward_split(corpus, purge=0)
        some = walk_forward_split(corpus, purge=3)
        assert some.n_purged > none.n_purged
        assert len(some.train) + len(some.val) + len(some.test) < len(corpus)

    def test_unsorted_input_is_still_split_by_time(self) -> None:
        """ordering must not depend on the corpus arriving pre-sorted."""
        batch = make_batch(n_episodes=100, n_steps=6)
        shuffled = batch.subset(np.random.default_rng(0).permutation(100))
        s = walk_forward_split(shuffled)
        assert s.train.open_epoch.max() < s.val.open_epoch.min()
        assert s.val.open_epoch.max() < s.test.open_epoch.min()

    def test_fractions_are_respected(self, corpus) -> None:
        s = walk_forward_split(corpus, train_frac=0.5, val_frac=0.25, purge=0)
        assert len(s.train) == pytest.approx(100, abs=2)
        assert len(s.val) == pytest.approx(50, abs=2)


class TestValidation:
    def test_rejects_tiny_corpus(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            walk_forward_split(make_batch(n_episodes=5, n_steps=4))

    @pytest.mark.parametrize("tr,va", [(0.9, 0.2), (1.0, 0.1), (0.0, 0.5)])
    def test_rejects_bad_fractions(self, corpus, tr, va) -> None:
        with pytest.raises(ValueError):
            walk_forward_split(corpus, train_frac=tr, val_frac=va)


class TestNormalizerWiring:
    """regression tests for a real bug: the normalizer is fit on the market
    block (9 features) but the observation is 13 wide. applying it to the whole
    vector is a shape error at best and a silent corruption at worst."""

    def test_fit_only_covers_market_features(self, corpus) -> None:
        s = walk_forward_split(corpus)
        assert s.normalizer.mean.shape == (len(MARKET_FEATURES),)
        assert s.normalizer.mean.shape != (N_FEATURES,)

    def test_env_accepts_the_fitted_normalizer(self, corpus) -> None:
        s = walk_forward_split(corpus)
        env = BinaryMarketEnv(s.train, normalizer=s.normalizer, random_episode_order=False)
        obs, _ = env.reset(seed=0, options={"episode": 0})
        assert obs.shape == (N_FEATURES,)
        assert np.all(np.isfinite(obs))

    def test_position_features_survive_normalisation_unchanged(self, corpus) -> None:
        """the trailing position block must pass through untransformed."""
        s = walk_forward_split(corpus)
        env = BinaryMarketEnv(
            s.train, normalizer=s.normalizer, max_position=100.0, random_episode_order=False
        )
        env.reset(seed=0, options={"episode": 0})
        obs, _, _, _, info = env.step(2)  # LONG

        n_market = len(MARKET_FEATURES)
        # position feature 0 is position / max_position, which must be exactly
        # 1.0 after going long, not a z-scored value.
        assert obs[n_market] == pytest.approx(1.0)
        assert info["position"] == pytest.approx(100.0)

    def test_normalised_market_block_is_roughly_centred(self, corpus) -> None:
        s = walk_forward_split(corpus)
        feats = s.train.market_features().reshape(-1, len(MARKET_FEATURES))
        z = s.normalizer.transform(feats)
        assert np.abs(z.mean(axis=0)).max() < 1e-6

    def test_unfitted_normalizer_raises_rather_than_passing_through(self, corpus) -> None:
        s = walk_forward_split(corpus, fit_normalizer=False)
        env = BinaryMarketEnv(s.train, normalizer=s.normalizer, random_episode_order=False)
        with pytest.raises(RuntimeError, match="before fit"):
            env.reset(seed=0, options={"episode": 0})


class TestNoLeakageAcrossSplits:
    def test_normalizer_stats_come_from_train_only(self) -> None:
        """shift the test period hard; train-fitted stats must not move."""
        n_ep, n_steps = 150, 8
        mid = np.full((n_ep, n_steps), 0.5, dtype=np.float32)
        mid[120:] = 0.95  # a violent regime change in the test period
        batch = make_batch(n_episodes=n_ep, n_steps=n_steps, mid_path=mid)

        shifted = walk_forward_split(batch)
        flat = walk_forward_split(make_batch(n_episodes=n_ep, n_steps=n_steps))

        # train periods are identical in both corpora, so the stats must be too
        np.testing.assert_allclose(
            shifted.normalizer.mean, flat.normalizer.mean, atol=1e-6
        )
