"""integrated gradients, as a second attribution family.

every attribution result in this project so far uses Shapley values. that
leaves an obvious objection: the failures shown might be properties of Shapley
estimation rather than of RL explanation. integrated gradients is a good
control because its mechanism is entirely different. Shapley perturbs
coalitions of features and averages marginal contributions; IG integrates the
model's gradient along a straight path from a baseline to the input. they share
no machinery.

they do share the structural property the null test needs. Shapley has
efficiency,

    sum_i phi_i = v(N) - v(empty)

and IG has completeness,

    sum_i phi_i = F(x) - F(baseline)

so both supply a scalar span for free, and the same test applies to either.

implemented from scratch against the definition in sundararajan et al. (2017).
the completeness axiom is asserted in tests/test_gradients.py rather than
assumed, since a wrong step count or a missing (x - baseline) factor produces
plausible-looking attributions that violate it.
"""

from __future__ import annotations

import numpy as np
import torch

from nano_rl.agents.networks import ActorCritic


def integrated_gradients(
    net: ActorCritic,
    x: np.ndarray,
    baseline: np.ndarray,
    action: int,
    n_steps: int = 128,
) -> np.ndarray:
    """attribution of pi(action | x) to each feature of x.

        phi_i = (x_i - b_i) * integral_0^1 d F(b + a(x - b)) / d x_i  d a

    the integral is a riemann sum over `n_steps` points on the straight path
    from baseline to input. more steps tighten completeness; 128 keeps the
    residual well under a percent on this network.

    args:
        net: the actor-critic whose policy is being explained.
        x: (n_features,) the input.
        baseline: (n_features,) the reference point. attribution is always
            relative to this, exactly as Shapley attribution is relative to a
            background distribution.
        action: which action's probability to attribute.
        n_steps: riemann steps.

    returns:
        (n_features,) attributions summing to F(x) - F(baseline).
    """
    x_t = torch.as_tensor(x, dtype=torch.float32)
    b_t = torch.as_tensor(baseline, dtype=torch.float32)

    # midpoint rule: alphas at the centres of n_steps equal intervals. this is
    # noticeably more accurate than the left rule at the same cost, which shows
    # up directly in the completeness residual.
    alphas = (torch.arange(n_steps, dtype=torch.float32) + 0.5) / n_steps
    path = b_t.unsqueeze(0) + alphas.unsqueeze(1) * (x_t - b_t).unsqueeze(0)
    path.requires_grad_(True)

    logits, _ = net(path)
    probs = torch.softmax(logits, dim=-1)[:, action].sum()
    grads = torch.autograd.grad(probs, path)[0]

    avg_grad = grads.mean(dim=0)
    return ((x_t - b_t) * avg_grad).detach().numpy()


def completeness_residual(
    net: ActorCritic,
    x: np.ndarray,
    baseline: np.ndarray,
    action: int,
    attributions: np.ndarray,
) -> float:
    """how far the attributions are from summing to F(x) - F(baseline).

    the IG analogue of the Shapley efficiency gap, and used the same way: a
    large residual means the approximation has not converged and the
    attributions should not be reported.
    """
    with torch.no_grad():
        both = torch.as_tensor(np.stack([x, baseline]), dtype=torch.float32)
        p = torch.softmax(net(both)[0], dim=-1)[:, action]
    return float(abs(attributions.sum() - (p[0] - p[1]).item()))


def ig_span(
    net: ActorCritic,
    states: np.ndarray,
    baseline: np.ndarray,
    n_steps: int = 128,
) -> float:
    """the IG analogue of the Shapley span, averaged over states.

    by completeness the attributions sum to F(x) - F(baseline), so that
    difference is the total amount the observation moves the policy toward its
    chosen action relative to the reference point. an agent that has learned
    nothing should barely move: its action is close to whatever the baseline
    already implied.

    the absolute value is taken per state before averaging. without it, states
    pushing in opposite directions cancel and a highly reactive policy scores
    the same as an inert one, which is the opposite of what the statistic is
    for.
    """
    with torch.no_grad():
        s_t = torch.as_tensor(states, dtype=torch.float32)
        b_t = torch.as_tensor(baseline, dtype=torch.float32).unsqueeze(0)
        probs_x = torch.softmax(net(s_t)[0], dim=-1)
        probs_b = torch.softmax(net(b_t)[0], dim=-1)[0]
        actions = probs_x.argmax(dim=-1)
        moved = probs_x.gather(1, actions.unsqueeze(1)).squeeze(1) - probs_b[actions]
    return float(moved.abs().mean().item())


def ig_attribution_profile(
    net: ActorCritic,
    states: np.ndarray,
    baseline: np.ndarray,
    n_steps: int = 64,
) -> np.ndarray:
    """mean absolute IG attribution per feature over a sample of states.

    the global summary, matching how the Shapley behaviour attributions are
    aggregated elsewhere, so the two families can be compared directly.
    """
    acc = np.zeros(states.shape[1])
    for row in states:
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(row, dtype=torch.float32).unsqueeze(0))
            a = int(logits.argmax().item())
        acc += np.abs(integrated_gradients(net, row, baseline, a, n_steps=n_steps))
    return acc / len(states)
