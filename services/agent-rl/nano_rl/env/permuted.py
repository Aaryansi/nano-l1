"""a null that keeps the agent sighted and the corpus fixed.

both null constructions this project has used are flawed in opposite
directions. training on synthetic signal-free corpora varies the corpus as well
as the information, which widens the reference and biases toward declining.
blinding the observation channel holds the corpus fixed but removes the agent's
capacity to respond to observational structure, so the null agents converge to
a constant policy, every null span is exactly zero, and the test fires on
anything.

what is wanted is the reinforcement-learning analogue of the label-permutation
test that Adebayo et al. use in supervised learning, which this paper claims to
be supplying and has so far approximated two different broken ways. The analogue
is direct once stated:

    permute the outcomes across episodes, and leave everything else alone.

the observation stream is untouched. every price path, every spread, every
volume and every spot feature is exactly what it was, because
EpisodeBatch.market_features is a function of the quote and spot arrays and
never reads `settlement`. an agent trained here sees a real market with real
structure and real frictions, can still learn to avoid costs, and simply cannot
predict the thing it is being paid to predict, because the outcome it is graded
against belongs to a different contract.

the marginal outcome rate is preserved exactly, since a permutation is a
relabelling. so the null agents face the same base rate as the observed agent
rather than an artificially balanced or skewed one.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nano_rl.env.binary_market import EpisodeBatch


def permute_outcomes(batch: EpisodeBatch, seed: int = 0) -> EpisodeBatch:
    """shuffle settlements across episodes, leaving observations identical.

    the cached market-feature block is carried over deliberately rather than
    invalidated. it does not depend on settlement, so recomputing it would cost
    time and produce the same array; passing it through also makes the
    invariant explicit, and tests assert the observations really are unchanged.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(batch))
    return replace(
        batch,
        settlement=np.asarray(batch.settlement)[order].copy(),
        _market_features=batch.market_features(),
    )


def outcome_rate(batch: EpisodeBatch) -> float:
    """fraction of episodes settling to 1, which permutation must preserve."""
    return float(np.mean(np.asarray(batch.settlement)))
