"""tests for the decision rule every verdict in the paper comes from.

nano_rl/explain/sanity.py decides, for each result reported, whether an
explanation is distinguishable from an explanation of nothing. it was the only
module in the library with no tests, which is the wrong module to leave
untested: it has already shipped one bug of exactly the kind that does not
announce itself, where a degenerate null returned z = 0.0 and silently turned
an overwhelming result into a null one.

the properties pinned here are the ones the paper's claims rest on, and each is
stated as the behaviour rather than as the implementation, so a rewrite that
preserves the semantics still passes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nano_rl.explain.sanity import SanityResult, consistency_across_runs
# aliased on import: pytest collects any module-level name beginning with
# "test_", so importing this one under its real name makes pytest try to run
# the function under test as if it were a test, and error on its arguments.
from nano_rl.explain.sanity import test_span_against_null as span_test


def spread(mean=0.0, sd=1.0, n=24, seed=0):
    """a non-degenerate null with a controlled mean and spread."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return list(mean + sd * (x - x.mean()) / x.std(ddof=1))


class TestTheBasicDecision:
    def test_an_extreme_observation_is_detected(self):
        r = span_test(50.0, spread())
        assert r.passes and r.z_score > 5

    def test_an_observation_inside_the_null_is_not(self):
        r = span_test(0.3, spread())
        assert not r.passes

    def test_the_statistic_is_reported_unchanged(self):
        r = span_test(7.25, spread())
        assert r.statistic == pytest.approx(7.25)

    def test_two_null_samples_is_the_minimum(self):
        with pytest.raises(ValueError):
            span_test(1.0, [0.0])
        span_test(1.0, [0.0, 0.1])  # must not raise


class TestBothCriteriaAreRequired:
    """the two-part rule, which is load bearing and not a formality.

    section 5.11 of the paper turns on a case where the normal statistic
    rejects and the rank statistic does not. if `passes` ever becomes either
    criterion alone, that section's conclusion silently flips.
    """

    def test_rank_alone_is_not_enough(self):
        # observation just outside a tight null: extreme in rank, weak in z
        nulls = spread(sd=1.0, n=4)
        r = span_test(max(nulls) + 0.01, nulls)
        assert r.p_rank <= r.min_achievable_p_rank + 1e-12
        assert r.p_normal >= 0.05
        assert not r.passes, "rank alone must not carry a rejection"

    def test_normal_alone_is_not_enough(self):
        # one far outlier inflates the sd enough that z is significant, while
        # the outlier itself is more extreme than the observation, so the rank
        # test declines
        nulls = [0.0] * 23 + [10.0]
        r = span_test(5.0, nulls)
        assert r.p_normal < 0.05
        assert r.p_rank > 0.05
        assert not r.passes, "the normal statistic alone must not carry it"

    def test_both_together_do_reject(self):
        r = span_test(50.0, spread())
        assert r.p_rank <= max(0.05, r.min_achievable_p_rank)
        assert r.p_normal < 0.05
        assert r.passes


class TestTheRankFloor:
    """with few null samples the rank p-value cannot reach 0.05 at all.

    this bit the project once: a signal 10.5 sd outside its null was reported
    as not significant because n = 8 puts the smallest achievable p at 0.111.
    the fix was to compare against max(alpha, floor), and it must stay.
    """

    @pytest.mark.parametrize("n,floor", [(4, 0.2), (8, 1 / 9), (24, 0.04)])
    def test_the_floor_is_one_over_n_plus_one(self, n, floor):
        r = span_test(1.0, spread(n=n))
        assert r.min_achievable_p_rank == pytest.approx(floor)

    def test_a_huge_signal_still_passes_with_a_small_null(self):
        r = span_test(500.0, spread(n=8))
        assert r.p_rank > 0.05, "the floor should make the raw p-value large"
        assert r.passes, "and the test must reject anyway"


class TestDegenerateNulls:
    """a zero-variance null. the case that already produced one shipped bug.

    the paper's position moved twice here: z = 0 (wrong, hides the result),
    then z = inf framed as maximally informative (wrong, no specificity), and
    now z = inf surfaced explicitly as a reason to distrust the reference.
    the arithmetic below is what all three versions disagreed about.
    """

    def test_it_is_flagged(self):
        assert span_test(5.0, [3.0] * 10).degenerate_null

    def test_a_deviation_gives_infinite_z_not_zero(self):
        r = span_test(5.0, [3.0] * 10)
        assert math.isinf(r.z_score) and r.z_score > 0
        assert r.z_score != 0.0, "the bug this replaced returned 0.0 here"

    def test_the_sign_follows_the_direction(self):
        assert span_test(1.0, [3.0] * 10).z_score == -math.inf

    def test_sitting_exactly_on_it_is_not_surprising(self):
        r = span_test(3.0, [3.0] * 10)
        assert r.z_score == 0.0
        assert not r.passes

    def test_infinite_z_does_not_produce_a_nan_p_value(self):
        r = span_test(5.0, [3.0] * 10)
        assert r.p_normal == 0.0 and not math.isnan(r.p_normal)


class TestSymmetryAndScale:
    def test_the_test_is_two_sided(self):
        """rejects below the null as well as above.

        worth pinning because it is the reason callers must check the SIGN of
        z before calling a rejection 'informative'. one script did not, and
        reported a span 2.8 sd below its null as evidence of information.
        """
        lo = span_test(-50.0, spread())
        hi = span_test(+50.0, spread())
        assert lo.passes and hi.passes
        assert lo.z_score < 0 < hi.z_score

    def test_shifting_everything_shifts_nothing(self):
        a = span_test(5.0, spread(mean=0.0))
        b = span_test(105.0, spread(mean=100.0))
        assert a.z_score == pytest.approx(b.z_score)
        assert a.passes == b.passes

    def test_rescaling_everything_leaves_z_alone(self):
        a = span_test(5.0, spread(sd=1.0))
        b = span_test(50.0, spread(sd=10.0))
        assert a.z_score == pytest.approx(b.z_score)


class TestReporting:
    def test_as_dict_carries_what_the_artifacts_need(self):
        d = span_test(50.0, spread()).as_dict()
        for k in ("statistic", "null_mean", "null_std", "p_rank", "p_normal",
                  "z_score", "passes", "n_null_samples"):
            assert k in d, f"artifacts and the verifier read {k}"
        assert d["n_null_samples"] == 24

    def test_summary_is_readable_and_mentions_the_verdict(self):
        s = span_test(50.0, spread()).summary()
        assert "z=" in s and ("informative" in s.lower()
                              or "distinguishable" in s.lower())

    def test_result_is_a_dataclass_with_the_null_retained(self):
        r = span_test(1.0, spread())
        assert isinstance(r, SanityResult)
        assert len(r.null_samples) == 24, "bootstrapping needs the raw draws"


class TestConsistencyAcrossRuns:
    def test_identical_runs_correlate_perfectly(self):
        v = np.array([3.0, -1.0, 2.0, 0.5])
        assert consistency_across_runs([v, v, v]) == pytest.approx(1.0)

    def test_it_ranks_by_magnitude_not_sign(self):
        a = np.array([3.0, -1.0, 2.0, 0.5])
        assert consistency_across_runs([a, -a]) == pytest.approx(1.0)

    def test_reversed_importance_anticorrelates(self):
        a = np.array([4.0, 3.0, 2.0, 1.0])
        assert consistency_across_runs([a, a[::-1]]) == pytest.approx(-1.0)

    def test_a_constant_run_is_skipped_rather_than_crashing(self):
        a = np.array([4.0, 3.0, 2.0, 1.0])
        flat = np.zeros(4)
        assert not math.isnan(consistency_across_runs([a, a, flat]))
