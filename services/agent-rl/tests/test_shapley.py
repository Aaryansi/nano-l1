"""tests for the shapley implementation.

this file matters more than most. the project's central claim in phase 5 is
that its attributions are correct, and an attribution method that is merely
plausible is worthless. so the implementation is checked three ways:

  1. against closed-form shapley values for a linear model, where
     phi_i = w_i * (x_i - E[x_i]) exactly.
  2. against the game-theoretic axioms: efficiency, symmetry, dummy, additivity.
  3. against the `shap` library's KernelExplainer, an independent
     implementation of the same quantity.

if all three agree, the code is right for reasons that do not depend on my own
reasoning about it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_rl.explain.shapley import (
    kernel_shap,
    masked_input_fn,
    permutation_shapley,
)


@pytest.fixture
def linear_setup():
    """a linear model, whose shapley values are known in closed form."""
    rng = np.random.default_rng(0)
    n_features = 6
    weights = np.array([3.0, -2.0, 0.0, 1.5, 0.0, 0.5])
    background = rng.normal(0.0, 1.0, size=(400, n_features))
    x = rng.normal(0.0, 1.0, size=n_features)

    def model_fn(z: np.ndarray) -> np.ndarray:
        return z @ weights

    # for a linear model with marginal (interventional) masking,
    # phi_i = w_i * (x_i - mean(background_i))
    expected = weights * (x - background.mean(axis=0))
    return x, background, model_fn, weights, expected, n_features


class TestAgainstClosedForm:
    def test_permutation_matches_linear_closed_form(self, linear_setup) -> None:
        x, bg, fn, _, expected, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = permutation_shapley(v, n, n_permutations=300, seed=0)
        np.testing.assert_allclose(att.values, expected, atol=0.02)

    def test_kernel_matches_linear_closed_form(self, linear_setup) -> None:
        x, bg, fn, _, expected, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = kernel_shap(v, n, n_samples=800, seed=0)
        np.testing.assert_allclose(att.values, expected, atol=0.02)

    def test_zero_weight_features_get_zero_attribution(self, linear_setup) -> None:
        """the dummy axiom, on a case where we know which features are dummies."""
        x, bg, fn, weights, _, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = permutation_shapley(v, n, n_permutations=300, seed=0)
        for i in np.where(weights == 0.0)[0]:
            assert abs(att.values[i]) < 0.02, f"dummy feature {i} got credit"

    def test_two_estimators_agree_with_each_other(self, linear_setup) -> None:
        x, bg, fn, _, _, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        a = permutation_shapley(v, n, n_permutations=300, seed=0)
        b = kernel_shap(v, n, n_samples=800, seed=0)
        np.testing.assert_allclose(a.values, b.values, atol=0.03)


class TestAxioms:
    """the four defining properties of the shapley value."""

    def test_efficiency(self, linear_setup) -> None:
        """values must sum to v(full) - v(empty)."""
        x, bg, fn, _, _, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = permutation_shapley(v, n, n_permutations=300, seed=0)
        assert att.relative_efficiency_gap < 0.02

    def test_kernel_shap_efficiency_is_exact(self, linear_setup) -> None:
        """imposed by substitution, so it must hold to machine precision."""
        x, bg, fn, _, _, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = kernel_shap(v, n, n_samples=200, seed=0)
        assert att.efficiency_gap < 1e-8

    def test_symmetry(self) -> None:
        """two features that contribute identically get identical values."""
        rng = np.random.default_rng(1)
        # centred exactly: under marginal masking phi_i = w_i*(x_i - mean_bg_i),
        # so identical weights alone do NOT imply identical attributions. the
        # reference means have to match as well, or the "asymmetry" is just
        # sampling noise in the background.
        bg = rng.normal(size=(300, 4))
        bg = bg - bg.mean(axis=0)
        x = np.array([1.0, 1.0, 0.5, -0.5])
        # features 0 and 1 enter with the same weight
        w = np.array([2.0, 2.0, 1.0, 1.0])

        v = masked_input_fn(x, bg, lambda z: z @ w, n_background=len(bg))
        att = permutation_shapley(v, 4, n_permutations=400, seed=0)
        assert att.values[0] == pytest.approx(att.values[1], abs=0.03)

    def test_additivity(self) -> None:
        """shapley of a sum of models is the sum of their shapley values."""
        rng = np.random.default_rng(2)
        bg = rng.normal(size=(300, 5))
        x = rng.normal(size=5)
        w1 = np.array([1.0, 0.0, 2.0, 0.0, -1.0])
        w2 = np.array([0.0, 3.0, -1.0, 0.5, 0.0])

        def att_of(w):
            v = masked_input_fn(x, bg, lambda z: z @ w, n_background=len(bg))
            return permutation_shapley(v, 5, n_permutations=300, seed=0).values

        combined = masked_input_fn(
            x, bg, lambda z: z @ (w1 + w2), n_background=len(bg)
        )
        got = permutation_shapley(combined, 5, n_permutations=300, seed=0).values
        np.testing.assert_allclose(got, att_of(w1) + att_of(w2), atol=0.03)


class TestNonlinear:
    def test_interaction_is_split_between_participants(self) -> None:
        """for f = x0 * x1 with zero-mean background, both must share credit.

        a method that credited only one of them would be a common and subtle
        bug, so it is pinned.
        """
        rng = np.random.default_rng(3)
        # centred so that E[x0] = E[x1] = 0 exactly, making the analytic answer
        # phi_0 = phi_1 = 2 rather than 2 plus background-mean noise.
        bg = rng.normal(0.0, 1.0, size=(600, 3))
        bg = bg - bg.mean(axis=0)
        x = np.array([2.0, 2.0, 0.0])

        def fn(z):
            return z[:, 0] * z[:, 1]

        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = permutation_shapley(v, 3, n_permutations=400, seed=0)

        # v({0,1}) = 4, v({0}) = v({1}) = 0, so phi_0 = phi_1 = 2 exactly
        assert att.values[0] == pytest.approx(2.0, abs=0.2)
        assert att.values[1] == pytest.approx(2.0, abs=0.2)
        assert abs(att.values[2]) < 0.2


class TestConvergence:
    def test_linear_model_has_zero_permutation_variance(self, linear_setup) -> None:
        """for a linear model every ordering gives the same marginal
        contribution, so the standard error is exactly zero. worth pinning,
        because it is why the convergence test below needs interactions."""
        x, bg, fn, _, _, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = permutation_shapley(v, n, n_permutations=50, seed=0)
        np.testing.assert_allclose(att.stderr, 0.0, atol=1e-12)

    def test_stderr_shrinks_with_more_permutations(self) -> None:
        """needs a model with interactions, where ordering actually matters."""
        rng = np.random.default_rng(11)
        bg = rng.normal(size=(200, 5))
        x = rng.normal(size=5)

        def fn(z):
            return z[:, 0] * z[:, 1] + z[:, 2] * z[:, 3]

        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        few = permutation_shapley(v, 5, n_permutations=20, seed=0)
        many = permutation_shapley(v, 5, n_permutations=400, seed=0)
        assert few.stderr.mean() > 0
        assert many.stderr.mean() < few.stderr.mean()

    def test_stderr_is_never_negative(self, linear_setup) -> None:
        x, bg, fn, _, _, n = linear_setup
        v = masked_input_fn(x, bg, fn, n_background=len(bg))
        att = permutation_shapley(v, n, n_permutations=50, seed=0)
        assert np.all(att.stderr >= 0)


class TestAgainstReferenceLibrary:
    """cross-validation against `shap`, an independent implementation.

    pinned in requirements.txt for this test alone. the attribution code in
    nano_rl/explain is written from scratch, not wrapped around it.
    """

    def test_matches_shap_kernel_explainer(self, linear_setup) -> None:
        shap = pytest.importorskip("shap")
        x, bg, fn, _, _, n = linear_setup

        # both explainers must be given the SAME background, or they are
        # answering different questions: under marginal masking the attribution
        # is defined relative to the reference distribution, and a different
        # subset has different column means.
        ref_bg = bg[:80]
        explainer = shap.KernelExplainer(fn, ref_bg)
        reference = explainer.shap_values(x, nsamples=600, silent=True)
        reference = np.asarray(reference).reshape(-1)

        v = masked_input_fn(x, ref_bg, fn, n_background=len(ref_bg), seed=0)
        ours = permutation_shapley(v, n, n_permutations=400, seed=0)

        np.testing.assert_allclose(ours.values, reference, atol=0.05)

    def test_matches_shap_on_a_nonlinear_model(self) -> None:
        shap = pytest.importorskip("shap")
        rng = np.random.default_rng(5)
        bg = rng.normal(size=(200, 4))
        x = rng.normal(size=4)

        def fn(z):
            return np.tanh(z[:, 0]) + z[:, 1] * z[:, 2]

        ref_bg = bg[:80]
        explainer = shap.KernelExplainer(fn, ref_bg)
        reference = np.asarray(
            explainer.shap_values(x, nsamples=800, silent=True)
        ).reshape(-1)

        v = masked_input_fn(x, ref_bg, fn, n_background=len(ref_bg), seed=0)
        ours = permutation_shapley(v, 4, n_permutations=500, seed=0)

        np.testing.assert_allclose(ours.values, reference, atol=0.08)


class TestAttributionContainer:
    def test_top_orders_by_absolute_value(self) -> None:
        from nano_rl.explain.shapley import Attribution

        att = Attribution(
            values=np.array([0.1, -5.0, 2.0]),
            stderr=np.zeros(3),
            base_value=0.0,
            full_value=-2.9,
            feature_names=("a", "b", "c"),
        )
        assert [n for n, _ in att.top(2)] == ["b", "c"]

    def test_efficiency_gap_detects_a_bad_estimate(self) -> None:
        from nano_rl.explain.shapley import Attribution

        att = Attribution(
            values=np.array([1.0, 1.0]),
            stderr=np.zeros(2),
            base_value=0.0,
            full_value=10.0,  # values sum to 2, not 10
            feature_names=("a", "b"),
        )
        assert att.efficiency_gap == pytest.approx(8.0)
