"""tests for evaluation metrics and baseline policies.

metrics get hand-computed expectations rather than golden values, because a
golden value recorded from a buggy implementation locks the bug in.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.baselines import (
    AlwaysFlat,
    BuyAndHold,
    LogisticBaseline,
    MeanReversion,
    RandomPolicy,
    default_baselines,
)
from nano_rl.env.features import N_FEATURES, feature_index
from nano_rl.metrics import (
    EPISODES_PER_YEAR,
    cluster_bootstrap_ci,
    cluster_bootstrap_p_value,
    compute_metrics,
    hit_rate,
    max_drawdown,
    paired_bootstrap_p_value,
    sharpe_ratio,
)

FLAT, LONG, SHORT = 1, 2, 0


class TestSharpe:
    def test_hand_computed(self) -> None:
        pnl = np.array([1.0, 2.0, 3.0])
        # mean 2, sample std 1
        assert sharpe_ratio(pnl) == pytest.approx(2.0)

    def test_zero_variance_returns_zero_not_inf(self) -> None:
        """a do-nothing policy must not report an infinite sharpe."""
        assert sharpe_ratio(np.zeros(50)) == 0.0
        assert sharpe_ratio(np.full(50, 3.0)) == 0.0

    def test_annualisation_uses_episode_count(self) -> None:
        pnl = np.array([1.0, 2.0, 3.0])
        expected = 2.0 * np.sqrt(EPISODES_PER_YEAR)
        assert sharpe_ratio(pnl, annualise=True) == pytest.approx(expected)

    def test_episodes_per_year_matches_15_minute_episodes(self) -> None:
        assert EPISODES_PER_YEAR == 365 * 24 * 4 == 35_040

    def test_too_short_series_is_zero(self) -> None:
        assert sharpe_ratio(np.array([1.0])) == 0.0


class TestMaxDrawdown:
    def test_monotonic_gain_has_no_drawdown(self) -> None:
        assert max_drawdown(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)

    def test_hand_computed(self) -> None:
        # cumulative: 10, 5, 8 -> peak 10, trough 5, drawdown 5
        assert max_drawdown(np.array([10.0, -5.0, 3.0])) == pytest.approx(5.0)

    def test_returned_positive(self) -> None:
        assert max_drawdown(np.array([-1.0, -1.0, -1.0])) >= 0

    def test_empty_is_zero(self) -> None:
        assert max_drawdown(np.array([])) == 0.0


class TestHitRate:
    def test_all_zeros_are_neither_wins_nor_losses(self) -> None:
        """a flat policy must not score a 100% hit rate."""
        hits, zeros = hit_rate(np.zeros(10))
        assert hits == 0.0
        assert zeros == 1.0

    def test_mixed(self) -> None:
        hits, zeros = hit_rate(np.array([1.0, -1.0, 0.0, 2.0]))
        assert hits == pytest.approx(0.5)
        assert zeros == pytest.approx(0.25)


class TestComputeMetrics:
    def test_assembles_consistently(self) -> None:
        pnl = np.array([1.0, -2.0, 3.0, 0.0])
        m = compute_metrics(pnl, np.array([1, 2, 1, 0]), np.array([0.1] * 4))
        assert m.n_episodes == 4
        assert m.total_pnl == pytest.approx(2.0)
        assert m.mean_pnl == pytest.approx(0.5)
        assert m.turnover == pytest.approx(m.mean_trades * 100.0)

    def test_flat_policy_row_is_all_zeros(self) -> None:
        m = compute_metrics(np.zeros(20), np.zeros(20), np.zeros(20))
        assert m.total_pnl == 0.0
        assert m.sharpe == 0.0
        assert m.max_drawdown == 0.0
        assert m.zero_rate == 1.0


class TestPairedBootstrap:
    def test_identical_series_is_not_significant(self) -> None:
        x = np.random.default_rng(0).normal(size=200)
        assert paired_bootstrap_p_value(x, x) == 1.0

    def test_large_consistent_difference_is_significant(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(5.0, 1.0, 300)
        b = a - 5.0
        assert paired_bootstrap_p_value(a, b) < 0.01

    def test_pure_noise_difference_is_not_significant(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(0, 50, 500)
        b = rng.normal(0, 50, 500)
        assert paired_bootstrap_p_value(a, b) > 0.05

    def test_mismatched_lengths_return_nan(self) -> None:
        assert np.isnan(paired_bootstrap_p_value(np.zeros(3), np.zeros(4)))


class TestBaselines:
    @pytest.fixture
    def obs(self) -> np.ndarray:
        return np.zeros(N_FEATURES)

    def test_always_flat_never_trades(self, obs) -> None:
        rng = np.random.default_rng(0)
        p = AlwaysFlat()
        assert all(p.act(obs, rng) == FLAT for _ in range(20))

    def test_buy_and_hold_always_long(self, obs) -> None:
        rng = np.random.default_rng(0)
        p = BuyAndHold()
        assert all(p.act(obs, rng) == LONG for _ in range(20))

    def test_random_covers_all_actions(self, obs) -> None:
        rng = np.random.default_rng(0)
        p = RandomPolicy()
        seen = {p.act(obs, rng) for _ in range(200)}
        assert seen == {0, 1, 2}

    def test_random_is_reproducible_from_the_seed(self, obs) -> None:
        p = RandomPolicy()
        a = [p.act(obs, np.random.default_rng(7)) for _ in range(5)]
        b = [p.act(obs, np.random.default_rng(7)) for _ in range(5)]
        assert a == b

    def test_mean_reversion_buys_a_dip(self) -> None:
        """price below its two-step-ago level should trigger a long."""
        obs = np.zeros(N_FEATURES)
        obs[feature_index("implied_prob")] = 0.50
        obs[feature_index("p_change_2")] = -0.05  # fell 0.05 over two steps
        p = MeanReversion()
        assert p.act(obs, np.random.default_rng(0)) == LONG

    def test_mean_reversion_sells_a_rip(self) -> None:
        obs = np.zeros(N_FEATURES)
        obs[feature_index("implied_prob")] = 0.50
        obs[feature_index("p_change_2")] = 0.05
        p = MeanReversion()
        assert p.act(obs, np.random.default_rng(0)) == SHORT

    def test_mean_reversion_guards_nonpositive_reference(self) -> None:
        obs = np.zeros(N_FEATURES)
        obs[feature_index("implied_prob")] = 0.0
        p = MeanReversion()
        assert p.act(obs, np.random.default_rng(0)) == FLAT

    def test_logistic_falls_back_to_momentum_without_a_model(self) -> None:
        p = LogisticBaseline(model=None)
        obs = np.zeros(N_FEATURES)
        obs[feature_index("implied_prob")] = 0.50
        obs[feature_index("p_change_1")] = 0.05  # strong up move
        # the original's fallback fades momentum
        assert p.act(obs, np.random.default_rng(0)) == SHORT

    def test_logistic_uses_the_trained_window_length(self) -> None:
        """regression guard: the shipped model keys on window_len.

        its largest coefficient by two orders of magnitude is on this feature,
        which was constant across its training set. passing the env's 14
        instead of the trained 50 flips its output.
        """
        assert LogisticBaseline().window_len == 50.0

    def test_default_set_has_the_required_baselines(self) -> None:
        names = {p.name for p in default_baselines()}
        assert {"always-flat", "buy-and-hold", "random", "mean-reversion"} <= names

    def test_refit_baseline_appears_only_when_supplied(self) -> None:
        assert len(default_baselines(logistic_refit=None)) == 5
        assert len(default_baselines(logistic_refit=object())) == 6

    def test_every_baseline_returns_a_legal_action(self) -> None:
        rng = np.random.default_rng(0)
        obs = np.zeros(N_FEATURES)
        for p in default_baselines():
            for _ in range(10):
                assert p.act(obs, rng) in (0, 1, 2)


class TestClusterBootstrap:
    """the seed is the unit of replication, not the episode.

    the steering experiment pools ~1,300 episodes from each of three seeds. a
    pooled paired bootstrap over all of them answers "how precisely do we know
    these three agents' mean difference", which is not the question the paper
    asks. the question is whether a fourth training run would show the same
    thing, and only seed-level resampling can speak to that.
    """

    @staticmethod
    def _one_odd_seed(rng):
        base = [rng.normal(0, 50, 1200) for _ in range(3)]
        eff = [
            base[0] + rng.normal(0.2, 50, 1200),
            base[1] + rng.normal(0.2, 50, 1200),
            base[2] + rng.normal(8.0, 50, 1200),
        ]
        return eff, base

    def test_pooling_is_anticonservative_when_seeds_disagree(self):
        rng = np.random.default_rng(0)
        eff, base = self._one_odd_seed(rng)
        pooled = paired_bootstrap_p_value(np.concatenate(eff), np.concatenate(base))
        cluster = cluster_bootstrap_p_value(eff, base)
        assert pooled < 0.01, "fixture should look significant when pooled"
        assert cluster > pooled * 10, (
            f"cluster p {cluster} should be far larger than pooled {pooled} "
            "when a single seed carries the effect"
        )

    def test_an_effect_present_in_every_seed_survives_clustering(self):
        rng = np.random.default_rng(1)
        base = [rng.normal(0, 50, 1200) for _ in range(5)]
        eff = [b + rng.normal(20.0, 50, 1200) for b in base]
        assert cluster_bootstrap_p_value(eff, base) < 0.05

    def test_no_difference_at_all_is_p_one(self):
        base = [np.arange(50, dtype=float) for _ in range(3)]
        assert cluster_bootstrap_p_value(base, list(base)) == 1.0

    def test_fewer_than_two_seeds_is_undefined(self):
        a = [np.arange(10, dtype=float)]
        assert np.isnan(cluster_bootstrap_p_value(a, [np.ones(10)]))

    def test_it_is_deterministic_given_a_seed(self):
        rng = np.random.default_rng(2)
        eff, base = self._one_odd_seed(rng)
        assert cluster_bootstrap_p_value(eff, base, seed=7) == \
               cluster_bootstrap_p_value(eff, base, seed=7)

    def test_it_is_a_probability(self):
        rng = np.random.default_rng(3)
        eff, base = self._one_odd_seed(rng)
        p = cluster_bootstrap_p_value(eff, base)
        assert 0.0 <= p <= 1.0


class TestClusterBootstrapInterval:
    """the interval a non-significant result needs.

    "no detected cost" is only informative alongside a bound on what the cost
    could have been. a p-value alone cannot distinguish "the effect is small"
    from "the experiment could not tell", which is the objection the steering
    claim has to answer.
    """

    def test_it_brackets_a_known_effect(self):
        rng = np.random.default_rng(0)
        base = [rng.normal(0, 50, 1200) for _ in range(4)]
        eff = [b + rng.normal(20.0, 50, 1200) for b in base]
        lo, hi = cluster_bootstrap_ci(eff, base)
        assert lo < 20.0 < hi, f"20 should lie inside [{lo}, {hi}]"

    def test_it_excludes_zero_when_the_effect_is_real(self):
        rng = np.random.default_rng(1)
        base = [rng.normal(0, 50, 1200) for _ in range(5)]
        eff = [b + rng.normal(25.0, 50, 1200) for b in base]
        lo, hi = cluster_bootstrap_ci(eff, base)
        assert lo > 0.0

    def test_it_contains_zero_when_there_is_no_effect(self):
        rng = np.random.default_rng(2)
        base = [rng.normal(0, 50, 1200) for _ in range(4)]
        same = [b + rng.normal(0.0, 50, 1200) for b in base]
        lo, hi = cluster_bootstrap_ci(same, base)
        assert lo < 0.0 < hi

    def test_it_agrees_with_the_p_value_it_accompanies(self):
        # if the interval excludes zero the p-value should reject, and vice
        # versa. they resample the same way, so disagreement means a bug.
        rng = np.random.default_rng(3)
        base = [rng.normal(0, 50, 1000) for _ in range(4)]
        for shift in (0.0, 5.0, 30.0):
            eff = [b + rng.normal(shift, 50, 1000) for b in base]
            lo, hi = cluster_bootstrap_ci(eff, base)
            p = cluster_bootstrap_p_value(eff, base)
            excludes_zero = lo > 0 or hi < 0
            assert excludes_zero == (p < 0.05), (
                f"shift={shift}: interval [{lo:.2f},{hi:.2f}] and p={p:.3f} disagree"
            )

    def test_fewer_than_two_seeds_is_undefined(self):
        lo, hi = cluster_bootstrap_ci([np.arange(10, dtype=float)], [np.ones(10)])
        assert np.isnan(lo) and np.isnan(hi)

    def test_the_interval_is_ordered(self):
        rng = np.random.default_rng(4)
        base = [rng.normal(0, 50, 900) for _ in range(3)]
        eff = [b + rng.normal(3.0, 50, 900) for b in base]
        lo, hi = cluster_bootstrap_ci(eff, base)
        assert lo <= hi
