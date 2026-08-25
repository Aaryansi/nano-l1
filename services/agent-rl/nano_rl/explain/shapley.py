"""shapley value attribution, implemented from scratch.

written directly against the definitions rather than wrapping a library. the
`shap` package is pinned in requirements.txt for one purpose only: to
cross-validate this implementation in tests/test_shapley.py. if the two
disagree, one of them is wrong and the test says so.

the shapley value of feature i for a characteristic function v is

    phi_i = sum over S subset of N\\{i} of
              |S|! (n-|S|-1)! / n!  *  [ v(S union {i}) - v(S) ]

which is the average marginal contribution of i over all orderings. exact
evaluation costs 2^n coalitions, so for n = 18 (262,144) we use two estimators:

  permutation sampling   unbiased, trivially correct, and it yields a standard
                         error for free. this is the default.

  kernel shap            solves a weighted least squares problem over sampled
                         coalitions. fewer characteristic-function evaluations
                         for the same accuracy, which matters when each one is
                         a full episode rollout.

the characteristic function is supplied by the caller. that is the whole design
point: the same estimator attributes a policy's action probability, a critic's
value, or an entire episode's return, depending on what v is. see
nano_rl/explain/trajectory.py for the return-based one.

references drawn on, not copied: shapley 1953 for the value; lundberg and lee
2017 for kernel shap's weighting kernel; castro et al. 2009 for permutation
sampling; beechey, smith and simsek 2023 (SVERL) for the argument that in rl
the characteristic function should be built on value, not on the policy output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# a characteristic function maps a boolean coalition mask to a scalar payoff.
# masks are (n_coalitions, n_features); the return is (n_coalitions,).
CharacteristicFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Attribution:
    """shapley values with uncertainty and a checkable efficiency residual."""

    values: np.ndarray  # (n_features,)
    stderr: np.ndarray  # (n_features,) zero when not estimated
    base_value: float  # v(empty set)
    full_value: float  # v(all features)
    feature_names: tuple[str, ...]

    @property
    def efficiency_gap(self) -> float:
        """how far the values are from summing to v(N) - v(empty).

        the shapley value is exactly efficient, so this is zero in theory. a
        large gap means the estimator has not converged, and reporting
        attributions without checking it is how noisy explanations get
        published as findings.
        """
        return float(abs(self.values.sum() - (self.full_value - self.base_value)))

    @property
    def relative_efficiency_gap(self) -> float:
        span = abs(self.full_value - self.base_value)
        return self.efficiency_gap / span if span > 1e-12 else 0.0

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        """the k features with the largest absolute attribution."""
        order = np.argsort(-np.abs(self.values))[:k]
        return [(self.feature_names[i], float(self.values[i])) for i in order]

    def as_dict(self) -> dict:
        return {
            "values": self.values.tolist(),
            "stderr": self.stderr.tolist(),
            "base_value": self.base_value,
            "full_value": self.full_value,
            "efficiency_gap": self.efficiency_gap,
            "feature_names": list(self.feature_names),
        }


def permutation_shapley(
    v: CharacteristicFn,
    n_features: int,
    n_permutations: int = 200,
    feature_names: tuple[str, ...] | None = None,
    seed: int = 0,
    batch: bool = True,
) -> Attribution:
    """unbiased shapley estimate by sampling orderings.

    for each sampled permutation, features are added one at a time and each
    feature is credited with the resulting change in v. averaging over
    permutations converges to the shapley value.

    args:
        v: characteristic function over boolean masks.
        n_features: n.
        n_permutations: sampled orderings. the standard error falls as
            1/sqrt(n_permutations).
        batch: evaluate a whole permutation's prefixes in one call to `v`.
            leave on when `v` is vectorised; turn off only for debugging.

    returns:
        an Attribution whose `stderr` is the standard error across
        permutations, so the caller can tell a real attribution from noise.
    """
    rng = np.random.default_rng(seed)
    contributions = np.zeros((n_permutations, n_features))

    for p in range(n_permutations):
        order = rng.permutation(n_features)

        # prefix masks: row j has the first j features of `order` switched on.
        # row 0 is the empty coalition, row n is the full one.
        masks = np.zeros((n_features + 1, n_features), dtype=bool)
        for j, feat in enumerate(order):
            masks[j + 1] = masks[j]
            masks[j + 1, feat] = True

        vals = v(masks) if batch else np.array([v(m[None, :])[0] for m in masks])
        deltas = np.diff(vals)
        contributions[p, order] = deltas

    values = contributions.mean(axis=0)
    stderr = contributions.std(axis=0, ddof=1) / np.sqrt(n_permutations)

    empty = v(np.zeros((1, n_features), dtype=bool))[0]
    full = v(np.ones((1, n_features), dtype=bool))[0]

    return Attribution(
        values=values,
        stderr=stderr,
        base_value=float(empty),
        full_value=float(full),
        feature_names=feature_names or tuple(f"f{i}" for i in range(n_features)),
    )


def _shapley_kernel_weight(n: int, s: int) -> float:
    """lundberg-lee kernel weight for a coalition of size s out of n.

    the endpoints s=0 and s=n carry infinite weight in the original
    formulation, since they pin the intercept and the efficiency constraint.
    we handle those as explicit constraints instead and return 0 here.
    """
    if s == 0 or s == n:
        return 0.0
    from math import comb

    return (n - 1) / (comb(n, s) * s * (n - s))


def kernel_shap(
    v: CharacteristicFn,
    n_features: int,
    n_samples: int = 512,
    feature_names: tuple[str, ...] | None = None,
    seed: int = 0,
) -> Attribution:
    """kernel shap: weighted least squares over sampled coalitions.

    cheaper than permutation sampling per unit of accuracy, which matters when
    each characteristic-function call is an episode rollout rather than a
    forward pass.

    the efficiency constraint sum(phi) = v(N) - v(empty) is imposed exactly, by
    substitution rather than as a penalty, so the returned values satisfy it to
    machine precision regardless of how few coalitions were sampled. that keeps
    the efficiency gap an honest diagnostic of the CALLER's characteristic
    function rather than of this solver.
    """
    rng = np.random.default_rng(seed)

    empty = float(v(np.zeros((1, n_features), dtype=bool))[0])
    full = float(v(np.ones((1, n_features), dtype=bool))[0])

    # sample coalition sizes by the kernel weight, then a uniform coalition of
    # that size. sizes near 1 and n-1 dominate, which is what makes kernel shap
    # sample-efficient.
    sizes = np.arange(1, n_features)
    weights = np.array([_shapley_kernel_weight(n_features, int(s)) for s in sizes])
    weights = weights / weights.sum()

    masks = np.zeros((n_samples, n_features), dtype=bool)
    for i in range(n_samples):
        s = rng.choice(sizes, p=weights)
        masks[i, rng.choice(n_features, size=int(s), replace=False)] = True

    y = v(masks) - empty

    # impose efficiency by eliminating the last coefficient:
    #   phi_last = (full - empty) - sum(phi_others)
    # so the design matrix becomes (m_i - m_last) on the remaining columns.
    z = masks.astype(float)
    total = full - empty
    x_reduced = z[:, :-1] - z[:, -1:]
    y_reduced = y - z[:, -1] * total

    # small ridge for numerical stability when coalitions are collinear
    a = x_reduced.T @ x_reduced + 1e-8 * np.eye(n_features - 1)
    b = x_reduced.T @ y_reduced
    phi_head = np.linalg.solve(a, b)

    values = np.append(phi_head, total - phi_head.sum())

    return Attribution(
        values=values,
        stderr=np.zeros(n_features),  # not estimated by this solver
        base_value=empty,
        full_value=full,
        feature_names=feature_names or tuple(f"f{i}" for i in range(n_features)),
    )


def masked_input_fn(
    x: np.ndarray,
    background: np.ndarray,
    model_fn: Callable[[np.ndarray], np.ndarray],
    n_background: int = 64,
    seed: int = 0,
) -> CharacteristicFn:
    """build a characteristic function for a single prediction.

    v(S) = E over background draws of model_fn(x with features outside S
    replaced by the background sample's values).

    this is the interventional / marginal formulation: absent features are
    replaced by draws from the data distribution rather than conditioned on.
    it is the standard choice and the one `shap`'s KernelExplainer uses, which
    is what makes the cross-validation test meaningful.

    args:
        x: (n_features,) the instance being explained.
        background: (n_bg, n_features) reference distribution.
        model_fn: maps (batch, n_features) -> (batch,) scalars.
        n_background: background draws averaged per coalition.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(background), size=min(n_background, len(background)),
                     replace=False)
    bg = background[idx]

    def v(masks: np.ndarray) -> np.ndarray:
        out = np.empty(len(masks))
        for i, m in enumerate(masks):
            # (n_bg, n_features): start from background, paste in the coalition
            synthetic = bg.copy()
            synthetic[:, m] = x[m]
            out[i] = float(np.mean(model_fn(synthetic)))
        return out

    return v
