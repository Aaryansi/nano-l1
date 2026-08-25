"""attribution targets for an rl agent, including a trajectory-aware one.

the phase-5 spec asks for explanations framed around three targets. they are
different questions and they have different answers, which is the point:

  BEHAVIOUR    what drives the action the policy takes at this state?
               characteristic function: pi(a* | s). this is ordinary shap on a
               classifier output.

  VALUE        what drives the critic's estimate of this state's worth?
               characteristic function: V(s). still a single-state question.

  OUTCOMES     what drives the return the agent actually earns?
               characteristic function: expected EPISODE RETURN when the agent
               observes only the features in the coalition, for the whole
               episode. this is the trajectory-aware one.

why the third is not redundant. the first two are one-step questions: they hold
the state fixed and ask what about it moved a single output. that silently
assumes the state is exogenous, which in a sequential problem it is not. the
agent's own earlier actions produced this state, and the feature that mattered
may have exerted its influence several steps ago and be invisible now.

the standard failure this produces is attributing to a PROXIMATE cause. an
agent holding a winning position will, at a mid-episode step, appear to be
acting on `position`, because that is what determines "keep holding". the
feature that actually earned the money is whatever told it to open the position
in the first place, and by mid-episode that feature may be doing nothing at
all. scripts/explain.py demonstrates exactly this case with ground truth.

the outcomes formulation follows the argument in beechey, smith and simsek
(SVERL, ICML 2023) that in rl the characteristic function should be built on
value or performance rather than on the policy's output. this implementation is
a simplified, from-scratch version of that idea: it is not SVERL, it does not
reproduce their estimators, and the limitations section of docs/REPORT.md says
so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nano_rl.agents.ppo import PPOAgent
from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.features import FEATURE_NAMES, N_FEATURES, FeatureNormalizer
from nano_rl.explain.shapley import (
    Attribution,
    kernel_shap,
    masked_input_fn,
    permutation_shapley,
)
from nano_rl.explain.rollout import VectorizedRollout


def _greedy_actions(agent: PPOAgent, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits, _ = agent.net(torch.as_tensor(obs, dtype=torch.float32))
        return logits.argmax(dim=-1).numpy()


def _action_prob(agent: PPOAgent, action: int):
    """model_fn returning the probability the policy assigns to `action`."""

    def fn(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probs = agent.net.action_probs(torch.as_tensor(z, dtype=torch.float32))
        return probs[:, action].numpy()

    return fn


def _value_fn(agent: PPOAgent):
    """model_fn returning the critic's value estimate."""

    def fn(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return agent.net.value(torch.as_tensor(z, dtype=torch.float32)).numpy()

    return fn


def explain_behaviour(
    agent: PPOAgent,
    obs: np.ndarray,
    background: np.ndarray,
    action: int | None = None,
    n_permutations: int = 200,
    seed: int = 0,
) -> tuple[Attribution, int]:
    """BEHAVIOUR target: attribute pi(a* | s) at a single state.

    returns (attribution, action explained). when `action` is None the agent's
    own greedy choice is explained, which is the question a reader actually has
    ("why did it do that?").
    """
    if action is None:
        action = int(_greedy_actions(agent, obs[None, :])[0])

    v = masked_input_fn(obs, background, _action_prob(agent, action), seed=seed)
    att = permutation_shapley(
        v, N_FEATURES, n_permutations=n_permutations,
        feature_names=FEATURE_NAMES, seed=seed,
    )
    return att, action


def explain_value(
    agent: PPOAgent,
    obs: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
) -> Attribution:
    """VALUE target: attribute the critic's V(s) at a single state."""
    v = masked_input_fn(obs, background, _value_fn(agent), seed=seed)
    return permutation_shapley(
        v, N_FEATURES, n_permutations=n_permutations,
        feature_names=FEATURE_NAMES, seed=seed,
    )


@dataclass
class OutcomeAttributionConfig:
    """knobs for the trajectory-aware estimator, which is the expensive one."""

    n_coalitions: int = 256
    n_episodes: int = 400
    seed: int = 0
    costs_enabled: bool = True


def explain_outcomes(
    agent: PPOAgent,
    batch: EpisodeBatch,
    background: np.ndarray,
    normalizer: FeatureNormalizer | None = None,
    max_position: float = 100.0,
    cfg: OutcomeAttributionConfig | None = None,
) -> Attribution:
    """OUTCOMES target: attribute the expected episode return.

    v(S) = mean episode return when the agent observes only the features in S
    for the WHOLE episode, with the rest replaced by background draws at every
    step.

    this is the trajectory-aware characteristic function. it costs one full
    batch replay per coalition, so kernel shap is used rather than permutation
    sampling: it needs far fewer characteristic-function evaluations for
    comparable accuracy, which is the binding constraint here.

    the efficiency identity is worth reading off the result. v(full) is the
    agent's actual mean return and v(empty) is its return while blind, so the
    attributions decompose exactly the value of having the observation at all.
    """
    cfg = cfg or OutcomeAttributionConfig()
    rng = np.random.default_rng(cfg.seed)

    sub = batch
    if cfg.n_episodes < len(batch):
        idx = np.arange(cfg.n_episodes)  # contiguous, to stay time-ordered
        sub = batch.subset(idx)

    roll = VectorizedRollout(
        sub,
        normalizer=normalizer,
        max_position=max_position,
        costs_enabled=cfg.costs_enabled,
    )

    def policy_for(mask: np.ndarray):
        def policy(obs: np.ndarray) -> np.ndarray:
            synthetic = obs.copy()
            if not mask.all():
                draws = background[rng.integers(0, len(background), size=len(obs))]
                synthetic[:, ~mask] = draws[:, ~mask]
            return _greedy_actions(agent, synthetic)

        return policy

    def v(masks: np.ndarray) -> np.ndarray:
        out = np.empty(len(masks))
        for i, m in enumerate(masks):
            out[i] = float(roll.run(policy_for(m))["returns"].mean())
        return out

    return kernel_shap(
        v,
        N_FEATURES,
        n_samples=cfg.n_coalitions,
        feature_names=FEATURE_NAMES,
        seed=cfg.seed,
    )


def compare_naive_and_trajectory(
    naive: Attribution, trajectory: Attribution, top_k: int = 5
) -> dict:
    """quantify how far the two explanations disagree.

    rank correlation is the headline: if the two orderings agree, the naive
    explanation is adequate and there is nothing to report. a low or negative
    correlation is the case the phase-5b spec asks for, and it needs to be
    shown rather than asserted.
    """
    a = np.abs(naive.values)
    b = np.abs(trajectory.values)

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(-x))
        return order.astype(float)

    # the degeneracy to guard is in the VALUES, not the ranks. ranks of n
    # distinct positions always have positive variance, so testing them would
    # report a perfect correlation between two attributions that are both
    # identically zero, i.e. claim two vacuous explanations agree.
    if a.max() < 1e-12 or b.max() < 1e-12:
        rho = float("nan")
    else:
        ra, rb = ranks(a), ranks(b)
        rho = float(np.corrcoef(ra, rb)[0, 1])

    top_a = {n for n, _ in naive.top(top_k)}
    top_b = {n for n, _ in trajectory.top(top_k)}

    return {
        "rank_correlation": rho,
        "top_k_overlap": len(top_a & top_b) / max(len(top_a | top_b), 1),
        "naive_top": naive.top(top_k),
        "trajectory_top": trajectory.top(top_k),
        "only_in_naive": sorted(top_a - top_b),
        "only_in_trajectory": sorted(top_b - top_a),
    }
