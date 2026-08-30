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
