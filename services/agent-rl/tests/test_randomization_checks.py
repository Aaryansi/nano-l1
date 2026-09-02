"""tests for the two checks added to answer a reviewer's objection.

both live in scripts/ because they are experiments, not library code, but the
pure functions inside them decide what the paper claims and so are pinned here.

  - parameter_randomization.randomize / similarity implement the canonical
    adebayo check. the paper's claim is that this check inverts our verdicts,
    which is only meaningful if the check is implemented faithfully: the
    randomization must actually change the weights, and the similarity metric
    must fall when the attributions disagree.

  - behavioural_equivalence.kl / js decide whether the steering result is
    stated as "not identified by behaviour" or the weaker and correct "not
    identified by task performance". a sign error in either would flip the
    paper's wording.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from behavioural_equivalence import js, kl  # noqa: E402
from parameter_randomization import LAYERS, randomize, similarity  # noqa: E402

from nano_rl.agents.networks import ActorCritic  # noqa: E402


class TestDivergences:
    def test_kl_of_a_distribution_with_itself_is_zero(self):
        p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        assert np.allclose(kl(p, p), 0.0, atol=1e-9)

    def test_kl_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.dirichlet(np.ones(3), size=8)
            q = rng.dirichlet(np.ones(3), size=8)
            assert (kl(p, q) >= -1e-12).all()

    def test_kl_is_not_symmetric_but_js_is(self):
        p = np.array([[0.7, 0.2, 0.1]])
        q = np.array([[0.2, 0.3, 0.5]])
        assert not np.allclose(kl(p, q), kl(q, p))
        assert np.allclose(js(p, q), js(q, p))

    def test_js_is_bounded_by_log_two(self):
        p = np.array([[1.0, 0.0, 0.0]])
        q = np.array([[0.0, 0.0, 1.0]])
        assert js(p, q) <= np.log(2.0) + 1e-9

    def test_js_grows_as_the_distributions_separate(self):
        p = np.array([[0.5, 0.5]])
        near = np.array([[0.45, 0.55]])
        far = np.array([[0.05, 0.95]])
        assert js(p, near) < js(p, far)

    def test_zero_probabilities_do_not_produce_nan(self):
        # clipping matters: an action the greedy policy never takes has
        # probability underflowing to zero, and log(0) would poison the mean.
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.5, 0.5]])
        assert np.isfinite(kl(p, q)).all()
        assert np.isfinite(kl(q, p)).all()


class TestSimilarity:
    def test_identical_attributions_are_perfectly_correlated(self):
        a = np.array([0.4, 0.1, 0.3, 0.2])
        rho, cos = similarity(a, a)
        assert rho == pytest.approx(1.0)
        assert cos == pytest.approx(1.0)

    def test_reversed_ranking_is_perfectly_anticorrelated(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        rho, _ = similarity(a, a[::-1].copy())
        assert rho == pytest.approx(-1.0)

    def test_a_constant_attribution_has_no_ranking_to_correlate(self):
        # spearman is undefined against a constant vector. returning nan is
        # what lets the caller average with nanmean instead of silently
        # recording a zero correlation that was never measured.
        a = np.array([0.25, 0.25, 0.25, 0.25])
        b = np.array([0.4, 0.1, 0.3, 0.2])
        rho, _ = similarity(a, b)
        assert np.isnan(rho)

    def test_cosine_ignores_scale_but_rank_ignores_it_too(self):
        a = np.array([0.4, 0.1, 0.3, 0.2])
        rho, cos = similarity(a, 10.0 * a)
        assert rho == pytest.approx(1.0)
        assert cos == pytest.approx(1.0)


class TestRandomize:
    def _net(self):
        torch.manual_seed(0)
        return ActorCritic(4, 3, 8)

    def test_it_changes_the_layers_it_is_given(self):
        net = self._net()
        before = net.policy_head.weight.detach().clone()
        randomize(net, ["policy_head"], seed=1)
        assert not torch.allclose(before, net.policy_head.weight)

    def test_it_leaves_the_other_layers_alone(self):
        net = self._net()
        before = net.trunk[0].weight.detach().clone()
        randomize(net, ["policy_head"], seed=1)
        assert torch.allclose(before, net.trunk[0].weight)

    def test_it_is_deterministic_given_the_seed(self):
        a, b = self._net(), self._net()
        randomize(a, LAYERS, seed=7)
        randomize(b, LAYERS, seed=7)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.allclose(pa, pb)

    def test_different_seeds_give_different_weights(self):
        a, b = self._net(), self._net()
        randomize(a, ["policy_head"], seed=1)
        randomize(b, ["policy_head"], seed=2)
        assert not torch.allclose(a.policy_head.weight, b.policy_head.weight)

    def test_cascading_randomizes_every_named_layer(self):
        net = self._net()
        before = [net.policy_head.weight.detach().clone(),
                  net.trunk[2].weight.detach().clone(),
                  net.trunk[0].weight.detach().clone()]
        randomize(net, LAYERS, seed=3)
        after = [net.policy_head.weight, net.trunk[2].weight, net.trunk[0].weight]
        for b_, a_ in zip(before, after):
            assert not torch.allclose(b_, a_)

    def test_the_policy_head_keeps_its_small_gain(self):
        # re-initialising with the network's own scheme is what makes the
        # comparison against "a net this architecture could have started from"
        # rather than against an arbitrary perturbation. the policy head's
        # gain is 0.01, so its weights stay much smaller than the trunk's.
        net = self._net()
        randomize(net, LAYERS, seed=11)
        assert net.policy_head.weight.abs().max() < net.trunk[0].weight.abs().max()

    def test_bias_is_reset_to_zero(self):
        net = self._net()
        with torch.no_grad():
            net.policy_head.bias.fill_(5.0)
        randomize(net, ["policy_head"], seed=1)
        assert torch.allclose(net.policy_head.bias, torch.zeros_like(net.policy_head.bias))
