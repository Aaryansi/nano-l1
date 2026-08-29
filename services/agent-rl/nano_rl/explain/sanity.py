"""a null-model test for whether an rl explanation carries any information.

the motivating observation, measured in scripts/stability.py and
scripts/evaluate.py: five independently trained agents produce highly
consistent explanations (rank correlation 0.865, 100% agreement on the single
most important feature) of a policy that earns -0.418 per episode and is
statistically indistinguishable from doing nothing (p = 0.13). the explanations
are stable, structured, and about nothing.

that matters because consistency across runs is widely used as a proxy for
trustworthiness. here it is high precisely where the explanation is empty, so
it cannot serve that role.

the test below is the reinforcement-learning analogue of the randomization
tests of adebayo et al. (2018), with one difference that is the point. in
supervised learning the null is artificial: randomize weights, or permute
labels. reinforcement learning admits a null that actually occurs in
deployment, namely an agent trained normally on real data whose environment
contains no exploitable structure. that is the null used here.

the test statistic comes from the shapley framework itself. by efficiency,

    sum_i phi_i = v(N) - v(empty)

so the SPAN v(N) - v(empty) is the total value of observing the state at all.
under the null it should be indistinguishable from zero, whatever the
individual attributions look like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nano_rl.explain.shapley import Attribution


@dataclass
class SanityResult:
    """outcome of testing one explanation against a null distribution."""

    statistic: float  # observed attribution span, v(N) - v(empty)
    null_mean: float
    null_std: float
    null_samples: list[float]
    p_rank: float  # two-sided, by null-distribution rank
    p_normal: float  # two-sided, normal approximation
    z_score: float
    passes: bool  # True when distinguishable from the null

    @property
    def degenerate_null(self) -> bool:
        """True when every null sample is identical.

        worth surfacing rather than hiding: a zero-variance null usually means
        the null agents are degenerate in some way (on Acrobot they fail
        completely with or without observations), and the reader should know
        the reference distribution has no spread before reading a p-value off
        it.
        """
        return bool(np.std(self.null_samples) <= 1e-12)

    @property
    def min_achievable_p_rank(self) -> float:
        """the smallest rank p-value this many null samples can produce.

        a rank test with n nulls bottoms out at 1/(n+1). with n=8 that is
        0.111, so the test cannot reject at 0.05 no matter how extreme the
        statistic. this was a real design error in an earlier run: a planted
        signal sitting 10.5 standard deviations outside the null still
        reported p=0.111 and was recorded as a failure to reject.
        """
        return 1.0 / (len(self.null_samples) + 1)

    def summary(self) -> str:
        verdict = "INFORMATIVE" if self.passes else "NOT DISTINGUISHABLE FROM NULL"
        floor = ""
        if self.p_rank <= self.min_achievable_p_rank + 1e-12:
            floor = f" (at the {self.min_achievable_p_rank:.3f} rank floor)"
        if self.degenerate_null:
            floor += " [degenerate null: zero variance]"
        return (
            f"span {self.statistic:+.4f} vs null {self.null_mean:+.4f} "
            f"+/- {self.null_std:.4f}  z={self.z_score:+.2f}  "
            f"p_rank={self.p_rank:.4f}{floor}  p_norm={self.p_normal:.2e}  "
            f"-> {verdict}"
        )

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "p_rank": self.p_rank,
            "p_normal": self.p_normal,
            "min_achievable_p_rank": self.min_achievable_p_rank,
            "z_score": self.z_score,
            "passes": self.passes,
            "n_null_samples": len(self.null_samples),
        }


def attribution_span(att: Attribution) -> float:
    """v(N) - v(empty): what the whole observation is worth.

    used rather than the sum of attributions because the two are equal by the
    efficiency axiom, and reading it off the endpoints avoids inheriting
    estimator noise from the individual values.
    """
    return float(att.full_value - att.base_value)


def test_span_against_null(
    stat: float,
    null_spans: np.ndarray | list[float],
    alpha: float = 0.05,
) -> SanityResult:
    """the test, taking the statistic and null spans directly.

    exists so a null distribution can be computed once and reused across many
    observed cases (a power sweep needs one null and dozens of observations).
    recomputing 24 null agents per sweep point would dominate the cost and
    would also introduce null-to-null variation that has nothing to do with
    the quantity being swept.
    """
    nulls = np.asarray(null_spans, dtype=float)
    if len(nulls) < 2:
        raise ValueError("need at least two null samples to form a distribution")

    mean, std = float(nulls.mean()), float(nulls.std(ddof=1))

    centred = np.abs(nulls - mean)
    p_rank = float((np.sum(centred >= abs(stat - mean)) + 1) / (len(nulls) + 1))

    # a degenerate null (every sample identical) is not "no information", it is
    # the opposite: any deviation from it is infinitely surprising. an earlier
    # version returned z = 0.0 here to avoid dividing by zero, which silently
    # converted an overwhelming result into a null one. that is the same
    # failure class as the rank-floor bug, and it was caught on Acrobot, where
    # blind agents score -500 whether or not they can see, so every null span
    # is exactly 0.
    from math import erfc, inf, isinf, sqrt

    if std > 1e-12:
        z = float((stat - mean) / std)
    elif abs(stat - mean) <= 1e-12:
        z = 0.0  # observation sits exactly on a degenerate null
    else:
        z = float(inf) if stat > mean else float(-inf)

    p_normal = 0.0 if isinf(z) else float(erfc(abs(z) / sqrt(2.0)))

    min_rank = 1.0 / (len(nulls) + 1)
    passes = (p_rank <= max(alpha, min_rank)) and (p_normal < alpha)

    return SanityResult(
        statistic=float(stat),
        null_mean=mean,
        null_std=std,
        null_samples=nulls.tolist(),
        p_rank=p_rank,
        p_normal=p_normal,
        z_score=z,
        passes=passes,
    )


def test_against_null(
    observed: Attribution,
    null_attributions: list[Attribution],
    alpha: float = 0.05,
) -> SanityResult:
    """is the observed explanation distinguishable from explanations of nothing?

    args:
        observed: attribution for the agent under scrutiny.
        null_attributions: attributions for matched agents trained on
            structure-free versions of the same environment.
        alpha: significance level.

    returns:
        a SanityResult. `passes` True means the explanation carries information
        the null cannot account for. False means the explanation is not
        distinguishable from one produced for an agent with nothing to learn,
        and should not be reported as a finding.

    two p-values are reported because neither alone is adequate here.

    the RANK p-value makes no distributional assumption, which is the right
    instinct for a return-based statistic. but it bottoms out at 1/(n+1), so
    with few null samples it cannot reject at any conventional level however
    extreme the observation. an earlier run with n=8 reported p=0.111 for a
    planted signal lying 10.5 standard deviations outside the null.

    the NORMAL p-value has resolution but assumes a shape the null need not
    have. `passes` requires BOTH, so the test is conservative: it will not
    declare an explanation informative on the strength of a distributional
    assumption alone, and it will not miss a ten-sigma effect for want of
    samples.
    """
    return test_span_against_null(
        attribution_span(observed),
        [attribution_span(a) for a in null_attributions],
        alpha=alpha,
    )


def certified_top_k(
    att: Attribution, k: int = 5, z_crit: float = 1.96
) -> dict:
    """is the top-k ranking separated beyond its own estimation error?

    this is the question RankSHAP and its relatives answer: given monte-carlo
    noise in the shapley estimates, can the top-k ordering be trusted? here it
    is checked by whether consecutive attributions are separated by more than
    the combined standard error of the pair.

    it is included to make a specific point, not as a contribution: this check
    can PASS on an explanation that the null test rejects. estimation
    certainty and explanatory validity are different properties, and
    controlling the first says nothing about the second. an explanation can be
    a precisely estimated description of nothing.
    """
    order = np.argsort(-np.abs(att.values))[:k]
    vals = np.abs(att.values)[order]
    errs = att.stderr[order]

    separated = []
    for i in range(len(order) - 1):
        gap = vals[i] - vals[i + 1]
        pooled = float(np.sqrt(errs[i] ** 2 + errs[i + 1] ** 2))
        separated.append(bool(gap > z_crit * pooled) if pooled > 0 else True)

    return {
        "features": [att.feature_names[i] for i in order],
        "values": [float(v) for v in vals],
        "stderr": [float(e) for e in errs],
        "adjacent_pairs_separated": separated,
        "fully_certified": all(separated) if separated else True,
    }


def consistency_across_runs(attributions: list[np.ndarray]) -> float:
    """mean pairwise rank correlation of |attribution| across runs.

    included so the paper's central claim can be checked in one place: this
    number is HIGH for the real agent, whose explanation the span test rejects.
    consistency and validity are different properties and this makes the gap
    between them measurable.
    """
    import itertools

    def ranks(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(-np.abs(x))).astype(float)

    rhos = []
    for i, j in itertools.combinations(range(len(attributions)), 2):
        ra, rb = ranks(attributions[i]), ranks(attributions[j])
        if ra.std() > 1e-12 and rb.std() > 1e-12:
            rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
    return float(np.mean(rhos)) if rhos else float("nan")
