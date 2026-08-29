"""tests for integrated gradients.

the completeness axiom is the load-bearing check. a wrong riemann step count,
a missing (x - baseline) factor, or gradients taken with respect to the wrong
tensor all produce attributions that look plausible and violate it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nano_rl.agents.networks import ActorCritic
from nano_rl.explain.gradients import (
    completeness_residual,
    ig_attribution_profile,
    ig_span,
    integrated_gradients,
)

N_FEAT, N_ACT = 6, 3


@pytest.fixture
def net() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(N_FEAT, N_ACT, hidden=32)


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    return rng.normal(size=N_FEAT), rng.normal(size=N_FEAT) * 0.1


class TestCompleteness:
    """sum of attributions must equal F(x) - F(baseline)."""

    def test_residual_is_small(self, net, points) -> None:
        x, b = points
        att = integrated_gradients(net, x, b, action=0, n_steps=256)
        assert completeness_residual(net, x, b, 0, att) < 1e-3

    def test_residual_shrinks_with_more_steps(self, net, points) -> None:
        x, b = points
        coarse = integrated_gradients(net, x, b, 0, n_steps=4)
        fine = integrated_gradients(net, x, b, 0, n_steps=512)
        assert completeness_residual(net, x, b, 0, fine) <= completeness_residual(
            net, x, b, 0, coarse
        )

    def test_holds_for_every_action(self, net, points) -> None:
        x, b = points
        for a in range(N_ACT):
            att = integrated_gradients(net, x, b, a, n_steps=256)
            assert completeness_residual(net, x, b, a, att) < 1e-3


class TestAxioms:
    def test_baseline_attributes_nothing(self, net, points) -> None:
        """explaining the baseline against itself must give exactly zero."""
        _, b = points
        att = integrated_gradients(net, b, b, action=0, n_steps=64)
        np.testing.assert_allclose(att, 0.0, atol=1e-9)

    def test_sensitivity_zero_for_unchanged_features(self, net, points) -> None:
        """a feature equal to its baseline value gets no attribution.

        this follows from the (x_i - b_i) factor, and its absence is the most
        common implementation bug.
        """
        x, b = points
        x = x.copy()
        x[2] = b[2]
        att = integrated_gradients(net, x, b, action=0, n_steps=128)
        assert abs(att[2]) < 1e-9

    def test_shape_and_finiteness(self, net, points) -> None:
        x, b = points
        att = integrated_gradients(net, x, b, 0, n_steps=32)
        assert att.shape == (N_FEAT,)
        assert np.all(np.isfinite(att))

    def test_deterministic(self, net, points) -> None:
        x, b = points
        a = integrated_gradients(net, x, b, 0, n_steps=64)
        c = integrated_gradients(net, x, b, 0, n_steps=64)
        np.testing.assert_array_equal(a, c)


class TestSpan:
    def test_span_is_non_negative(self, net) -> None:
        rng = np.random.default_rng(1)
        states = rng.normal(size=(40, N_FEAT))
        b = np.zeros(N_FEAT)
        assert ig_span(net, states, b) >= 0.0

    def test_constant_policy_has_near_zero_span(self) -> None:
        """a net whose output ignores its input should barely move from baseline."""
        torch.manual_seed(3)
        net = ActorCritic(N_FEAT, N_ACT, hidden=16)
        # zero the trunk so the policy is constant in the input
        with torch.no_grad():
            for p in net.trunk.parameters():
                p.zero_()
        rng = np.random.default_rng(2)
        states = rng.normal(size=(40, N_FEAT))
        assert ig_span(net, states, np.zeros(N_FEAT)) < 1e-6

    def test_responsive_policy_has_larger_span_than_constant(self, net) -> None:
        rng = np.random.default_rng(4)
        states = rng.normal(size=(60, N_FEAT)) * 3.0
        b = np.zeros(N_FEAT)

        torch.manual_seed(5)
        flat = ActorCritic(N_FEAT, N_ACT, hidden=16)
        with torch.no_grad():
            for p in flat.trunk.parameters():
                p.zero_()

        assert ig_span(net, states, b) > ig_span(flat, states, b)


class TestProfile:
    def test_profile_shape_and_non_negative(self, net) -> None:
        rng = np.random.default_rng(6)
        states = rng.normal(size=(15, N_FEAT))
        prof = ig_attribution_profile(net, states, np.zeros(N_FEAT), n_steps=16)
        assert prof.shape == (N_FEAT,)
        assert np.all(prof >= 0)

    def test_profile_is_not_uniform_for_a_real_net(self, net) -> None:
        """a trained-ish network should weight features differently."""
        rng = np.random.default_rng(7)
        states = rng.normal(size=(25, N_FEAT))
        prof = ig_attribution_profile(net, states, np.zeros(N_FEAT), n_steps=32)
        assert prof.std() > 1e-6
