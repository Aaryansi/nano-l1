"""tests for the outcome-permutation null.

the construction's entire claim is that it removes the information and nothing
else. that is a statement about which arrays change, so it is asserted directly
rather than inferred from downstream results: if permutation ever perturbed an
observation, the null would differ from the observed agent's world for a second
reason and the comparison would be confounded exactly the way the two rejected
constructions were.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.env.permuted import outcome_rate, permute_outcomes
from nano_rl.env.synthetic import make_learnable_corpus


@pytest.fixture(scope="module")
def corpus():
    return make_learnable_corpus(n_episodes=120, n_steps=8, seed=0)


class TestOnlyTheOutcomeMoves:
    def test_observations_are_bit_identical(self, corpus):
        p = permute_outcomes(corpus, seed=1)
        np.testing.assert_array_equal(
            corpus.market_features(), p.market_features()
        )

    @pytest.mark.parametrize(
        "field", ["bid", "ask", "last_price", "volume", "staleness",
                  "flow_imbalance", "t_sec", "open_epoch"]
    )
    def test_every_other_array_is_untouched(self, corpus, field):
        p = permute_outcomes(corpus, seed=1)
        np.testing.assert_array_equal(getattr(corpus, field), getattr(p, field))

    def test_the_settlement_actually_moves(self, corpus):
        p = permute_outcomes(corpus, seed=1)
        assert not np.array_equal(corpus.settlement, p.settlement)

    def test_the_source_batch_is_not_mutated(self, corpus):
        before = np.array(corpus.settlement, copy=True)
        permute_outcomes(corpus, seed=2)
        np.testing.assert_array_equal(corpus.settlement, before)


class TestItIsAPermutation:
    def test_outcome_rate_is_preserved_exactly(self, corpus):
        p = permute_outcomes(corpus, seed=3)
        assert outcome_rate(p) == pytest.approx(outcome_rate(corpus))

    def test_it_is_a_relabelling_not_a_resample(self, corpus):
        """multiset equality: the same outcomes, in a different order.

        a resample would preserve the rate only in expectation and would let a
        null draw be, say, all ones, which is a different environment rather
        than the same one relabelled.
        """
        p = permute_outcomes(corpus, seed=4)
        np.testing.assert_array_equal(
            np.sort(corpus.settlement), np.sort(p.settlement)
        )

    def test_the_seed_controls_the_permutation(self, corpus):
        a = permute_outcomes(corpus, seed=5)
        b = permute_outcomes(corpus, seed=5)
        c = permute_outcomes(corpus, seed=6)
        np.testing.assert_array_equal(a.settlement, b.settlement)
        assert not np.array_equal(a.settlement, c.settlement)


class TestItDestroysThePlantedSignal:
    def test_the_informative_feature_stops_predicting(self, corpus):
        """the point of the construction, measured rather than assumed.

        on a planted corpus some feature correlates with the outcome. after
        permutation no feature may, or the null agents would still have
        something to learn and the reference would be too generous.
        """
        feats = corpus.market_features().mean(axis=1)  # (episodes, features)
        p = permute_outcomes(corpus, seed=7)

        def best_abs_corr(settle):
            y = np.asarray(settle, dtype=float)
            out = 0.0
            for j in range(feats.shape[1]):
                x = feats[:, j]
                if x.std() < 1e-12:
                    continue
                out = max(out, abs(float(np.corrcoef(x, y)[0, 1])))
            return out

        before = best_abs_corr(corpus.settlement)
        after = best_abs_corr(p.settlement)
        assert before > 0.15, "planted corpus should carry a usable signal"
        assert after < before / 2.0


class TestStratifiedPermutation:
    """the variant that preserves calibration by only shuffling like with like."""

    def test_observations_still_untouched(self, corpus):
        from nano_rl.env.permuted import permute_outcomes_stratified
        p = permute_outcomes_stratified(corpus, n_buckets=4, seed=1)
        np.testing.assert_array_equal(
            corpus.market_features(), p.market_features()
        )

    def test_still_a_permutation(self, corpus):
        from nano_rl.env.permuted import permute_outcomes_stratified
        p = permute_outcomes_stratified(corpus, n_buckets=4, seed=1)
        np.testing.assert_array_equal(
            np.sort(corpus.settlement), np.sort(p.settlement)
        )

    def test_labels_stay_inside_their_bucket(self, corpus):
        """the whole point: a contract only swaps with similarly priced ones.

        checked by bucket-mean rather than by tracking indices, because
        preserving each bucket's base rate is the property that preserves
        calibration, and it is the property that matters.
        """
        from nano_rl.env.permuted import permute_outcomes_stratified
        n = 4
        p = permute_outcomes_stratified(corpus, n_buckets=n, seed=2)
        last = 0.5 * (np.asarray(corpus.bid)[:, -1] + np.asarray(corpus.ask)[:, -1])
        edges = np.quantile(last, np.linspace(0.0, 1.0, n + 1))
        bucket = np.digitize(last, np.unique(edges)[1:-1], right=False)
        for b in np.unique(bucket):
            m = bucket == b
            assert np.asarray(corpus.settlement, float)[m].mean() == pytest.approx(
                np.asarray(p.settlement, float)[m].mean()
            )

    def test_it_removes_less_than_the_plain_permutation(self):
        """the finding, as a test: stratifying costs you removal.

        needs a corpus whose terminal price actually predicts the outcome. the
        module fixture's price is constant at the last step, so every episode
        lands in one bucket and stratifying is a no-op; that is a property of
        the fixture, not of the construction, and asserting on it would pin the
        wrong thing.

        averaged over seeds because a single permutation can go either way by
        chance.
        """
        from dataclasses import replace

        from nano_rl.env.permuted import (
            permute_outcomes,
            permute_outcomes_stratified,
        )

        base = make_learnable_corpus(n_episodes=400, n_steps=8, seed=0)
        y = np.asarray(base.settlement, dtype=float)
        rng = np.random.default_rng(0)
        # a terminal quote that mostly agrees with the outcome, which is what
        # gives stratification something to stratify on
        price = np.where(y > 0.5, 0.95, 0.05) + rng.normal(0, 0.02, size=len(y))
        bid, ask = np.asarray(base.bid).copy(), np.asarray(base.ask).copy()
        bid[:, -1], ask[:, -1] = price - 0.01, price + 0.01
        corpus = replace(base, bid=bid, ask=ask, _market_features=None)

        plain, strat = [], []
        for seed in range(20):
            plain.append((y != np.asarray(
                permute_outcomes(corpus, seed).settlement, float)).mean())
            strat.append((y != np.asarray(
                permute_outcomes_stratified(corpus, 8, seed).settlement,
                float)).mean())
        assert np.mean(strat) < np.mean(plain)
