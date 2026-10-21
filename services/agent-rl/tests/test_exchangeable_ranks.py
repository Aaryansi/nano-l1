"""Enumerate all possible observed indices, rather than simulating nominal size."""
from itertools import combinations_with_replacement

import numpy as np
import pytest

from nano_rl.explain.rank import exchangeable_rank_p, rank_comparisons
from nano_rl.explain.sanity import test_span_against_null as legacy


@pytest.mark.parametrize("alternative", ["greater", "two-sided"])
def test_finite_pool_rank_is_superuniform_with_ties(alternative):
    for n in range(2, 8):
        for values in combinations_with_replacement([-2., 0., 3.], n):
            pool = np.array(values)
            ps = [exchangeable_rank_p(pool[i], np.delete(pool, i), alternative=alternative)
                  for i in range(n)]
            for alpha in [.01, .05, .1, .25, .5, .9]:
                assert sum(p <= alpha for p in ps) <= int(np.floor(alpha * n + 1e-12))


def test_legacy_reference_center_can_double_nominal_exchangeable_size():
    # This joint law designates one index uniformly in a fixed pool. It is
    # exchangeable, not a sample from the synthetic RL benchmark's iid DGP.
    pool = np.array([-1., 1.] + [0.] * 18)
    old, symmetric, upper = [], [], []
    for i, stat in enumerate(pool):
        refs = np.delete(pool, i)
        old.append(legacy(stat, refs).passes)
        symmetric.append(exchangeable_rank_p(stat, refs, alternative="two-sided"))
        upper.append(exchangeable_rank_p(stat, refs))
    assert sum(old) == 2
    assert symmetric[:2] == [.1, .1]
    assert sum(p <= .05 for p in symmetric) == 0
    assert sum(p <= .05 for p in upper) == 1


@pytest.mark.parametrize("alternative", ["greater", "two-sided"])
def test_order_invariance_and_all_ties(alternative):
    refs = np.array([-1e308, 2., 0., 1e308, -20.])
    p = exchangeable_rank_p(1., refs, alternative=alternative)
    assert exchangeable_rank_p(1., refs[::-1], alternative=alternative) == p
    assert exchangeable_rank_p(2., [2.] * 24, alternative=alternative) == 1.


def test_subfloor_rank_is_computed_but_cannot_reject():
    result = rank_comparisons(100., [0., 1.])
    for row in result.values():
        assert row["rank_floor"] == 1 / 3
        assert not row["rank_floor_at_or_below_alpha"]
        assert not row["rejected"]


@pytest.mark.parametrize("stat,refs", [(np.nan, [0., 1.]), (1., [0., np.inf]),
                                      (1., []), (1., [[0., 1.]])])
def test_rank_rejects_invalid_inputs(stat, refs):
    with pytest.raises(ValueError):
        exchangeable_rank_p(stat, refs)
    with pytest.raises(ValueError):
        legacy(stat, refs)


@pytest.mark.parametrize("alpha", [0., 1., np.nan])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        rank_comparisons(1., [0., 1.], alpha=alpha)
    with pytest.raises(ValueError):
        legacy(1., [0., 1.], alpha=alpha)


def test_legacy_scale_flag_matches_its_decision_scale():
    result = legacy(1., [0., 1.5e-12])
    assert result.null_std > 1e-12
    assert not result.degenerate_null
