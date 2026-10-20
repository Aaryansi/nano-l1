"""Diagnostics must not manufacture stability from ties or absolute values."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from nano_rl.explain.diagnostics import magnitude_rank_correlation
from nano_rl.explain.sanity import ranking_gap_diagnostic, consistency_across_runs
from nano_rl.explain.shapley import Attribution

SPEC = importlib.util.spec_from_file_location("stability", Path(__file__).resolve().parents[1] / "scripts/stability.py")
stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stability)


def test_tied_rank_correlation_matches_independent_reference():
    a = np.array([1., 0., 0., -3., 0.])
    b = np.array([0., 2., 0., 3., 0.])
    assert magnitude_rank_correlation(a, b) == pytest.approx(spearmanr(abs(a), abs(b)).statistic)
    permutation = [4, 2, 0, 3, 1]
    assert magnitude_rank_correlation(a[permutation], b[permutation]) == pytest.approx(magnitude_rank_correlation(a, b))


def test_constant_attributions_have_no_rank_information():
    assert np.isnan(magnitude_rank_correlation(np.zeros(4), np.arange(4)))
    assert np.isnan(consistency_across_runs([np.zeros(4), np.zeros(4)]))
    row = stability.summarise("empty", [np.zeros(4), np.zeros(4)])
    assert row["n_defined_rank_pairs"] == 0
    assert row["n_unique_top_pairs"] == 0
    assert np.isnan(row["top1_agreement"])


def test_opposite_signed_means_are_not_perfect_sign_agreement():
    magnitude = np.array([1., 2., 3.])
    row = stability.summarise("behavior", [magnitude, magnitude],
                              [magnitude, -magnitude])
    assert row["rank_corr_mean"] == pytest.approx(1.)
    assert row["sign_agreement"] == 0.


def test_top_k_diagnostic_checks_selection_boundary_and_is_not_a_certificate():
    att = Attribution(values=np.array([5., 4., 3., 2.99]), stderr=np.full(4, .1),
                      base_value=0., full_value=14.99, feature_names=("a", "b", "c", "d"))
    result = ranking_gap_diagnostic(att, k=3)
    assert result["adjacent_pairs_separated"] == [True, True, False]
    assert result["includes_boundary"]
    assert not result["all_gaps_separated"]
    assert not result["is_formal_certificate"]


def test_exactly_tied_zero_error_estimates_are_not_separated():
    att = Attribution(values=np.array([1., 1.]), stderr=np.zeros(2),
                      base_value=0., full_value=2., feature_names=("a", "b"))
    assert not ranking_gap_diagnostic(att, k=1)["all_gaps_separated"]
