"""Synthetic reference fixtures for training and explanation checks.

The planted population supplies repeated noisy settlement signals. Its
first-observation sign strategy is a specified reference, not a proven optimum.
At the default balanced price, the zero-strength population has no predictive
feature signal and flat earns zero population return before finite-sample noise.
Neither low market correlations nor a failed learner establish market efficiency.
All fixtures use the actual EpisodeBatch accounting and assumed cost model.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.costs import fee_dollars
from nano_rl.env.features import N_SPOT

# index of spot_ret_since_open within the spot block. the signal is injected
# here because it is the semantically honest place: for a "BTC up?" contract,
# the underlying's move genuinely should predict resolution.
SIGNAL_IDX = 0


@dataclass(frozen=True)
class SyntheticTrainEval:
    train: EpisodeBatch
    evaluation: EpisodeBatch
    train_seed: int
    evaluation_seed: int

    def metadata(self) -> dict:
        return {"train_seed": self.train_seed, "evaluation_seed": self.evaluation_seed,
                "train_episodes": len(self.train), "evaluation_episodes": len(self.evaluation)}


def synthetic_train_eval(factory, *, seed: int, **kwargs) -> SyntheticTrainEval:
    """Independent corpus draws from one specified population distribution."""
    evaluation_seed = seed + 1_000_000
    return SyntheticTrainEval(factory(seed=seed, **kwargs),
                              factory(seed=evaluation_seed, **kwargs), seed, evaluation_seed)


def make_learnable_corpus(
    n_episodes: int = 2000,
    n_steps: int = 14,
    signal_strength: float = 1.0,
    noise: float = 0.3,
    price: float = 0.5,
    spread: float = 0.02,
    seed: int = 0,
) -> EpisodeBatch:
    """a corpus where one observed feature predicts settlement.

    construction:
        settlement ~ Bernoulli(1/2)
        signal     = (2*settlement - 1) * signal_strength + N(0, noise)
        mid price  = `price`, constant and therefore uninformative

    Repeated noisy signals can support aggregation, delayed entry or abstention.
    The first-observation sign policy is a transparent reference strategy;
    its realized corpus payoff is not a proven optimal population value.

    args:
        signal_strength: separation between the two outcome classes. 1.0 with
            noise 0.3 gives a very learnable but not trivial problem.
        noise: gaussian noise on the signal.
        price: constant mid. 0.5 puts the fee at its maximum, which is the
            maximum for the assumed coefficient.

    returns:
        an EpisodeBatch whose `spot` block carries the signal.
    """
    rng = np.random.default_rng(seed)

    settlement = rng.integers(0, 2, size=n_episodes).astype(np.float32)
    sign = 2.0 * settlement - 1.0  # -1 or +1

    # the signal is constant within an episode plus per-step noise, so the
    # agent can act on it from the first step.
    base = sign[:, None] * signal_strength
    signal = base + rng.normal(0.0, noise, size=(n_episodes, n_steps))

    spot = np.zeros((n_episodes, n_steps, N_SPOT), dtype=np.float32)
    spot[:, :, SIGNAL_IDX] = signal.astype(np.float32)

    half = spread / 2.0
    mid = np.full((n_episodes, n_steps), price, dtype=np.float32)

    return EpisodeBatch(
        bid=(mid - half).astype(np.float32),
        ask=(mid + half).astype(np.float32),
        last_price=mid.copy(),
        volume=np.full((n_episodes, n_steps), 10_000.0, dtype=np.float32),
        staleness=np.zeros((n_episodes, n_steps), dtype=np.float32),
        flow_imbalance=np.zeros((n_episodes, n_steps), dtype=np.float32),
        t_sec=np.tile(
            np.arange(1, n_steps + 1, dtype=np.float32) * 60.0, (n_episodes, 1)
        ),
        settlement=settlement,
        open_epoch=np.arange(n_episodes, dtype=np.float64) * 900.0,
        spot=spot,
    )


def make_null_corpus(
    n_episodes: int = 2000,
    n_steps: int = 14,
    noise: float = 0.3,
    price: float = 0.5,
    spread: float = 0.02,
    seed: int = 0,
) -> EpisodeBatch:
    """same shape and frictions, but the feature carries no information.

    At the default price=0.5 and balanced independent settlement population,
    a predictable trading strategy has no positive population edge before
    costs. Flat earns exactly zero. Other prices can create a directional
    base-rate edge even though the feature signal strength remains zero.
    """
    batch = make_learnable_corpus(
        n_episodes=n_episodes,
        n_steps=n_steps,
        signal_strength=0.0,  # signal is pure noise
        noise=noise,
        price=price,
        spread=spread,
        seed=seed,
    )
    return batch


def signal_policy_return(
    batch: EpisodeBatch,
    max_position: float = 100.0,
    with_costs: bool = True,
) -> float:
    """mean per-episode return of "read the signal once, hold to settlement".

    This computes the exact realized payoff of that fixed strategy on the
    supplied corpus under the assumed entry fees. It is not a proof of
    optimality: repeated noisy signals, waiting and abstention can matter.
    On the default balanced zero-strength population, following the first
    noise draw has zero expected gross payoff and incurs transaction costs.
    """
    signal = batch.spot[:, 0, SIGNAL_IDX]  # first-step signal
    action = np.sign(signal)  # +1 long, -1 short
    settlement = batch.settlement

    entry = batch.ask[:, 0]  # buying lifts the ask
    exit_short = batch.bid[:, 0]  # selling hits the bid
    price = np.where(action > 0, entry, exit_short)

    # payoff per contract: long pays (settlement - entry), short pays
    # (entry - settlement).
    per_contract = np.where(
        action > 0, settlement - price, price - settlement
    )
    gross = per_contract * max_position

    if with_costs:
        fees = np.array([fee_dollars(max_position, float(p)) for p in price])
        gross = gross - fees

    # a zero signal means no position and therefore no pnl
    gross = np.where(action == 0, 0.0, gross)
    return float(gross.mean())


def discretize(obs: np.ndarray, signal_idx: int, n_bins: int = 9) -> int:
    """map an observation to a tabular state index.

    used only by the tabular-q sanity check. bins the signal feature and
    ignores everything else, which is sufficient because the synthetic corpora
    put all the information in that one feature.
    """
    edges = np.linspace(-2.0, 2.0, n_bins - 1)
    return int(np.digitize(obs[signal_idx], edges))


def flat_policy_return() -> float:
    """return of always staying flat: exactly zero, by construction.

    trivial, but named so that the null corpus has an explicit benchmark
    rather than being compared against `signal_policy_return`, which is the
    wrong target there.
    """
    return 0.0


# index of the decoy feature within the spot block. `spot_implied_gap` is
# reused as the carrier so the observation width stays at N_FEATURES.
DECOY_IDX = 4


def make_decoy_corpus(
    n_episodes: int = 3000,
    n_steps: int = 14,
    train_frac: float = 0.6,
    decoy_strength: float = 2.0,
    signal_strength: float = 0.0,
    noise: float = 0.3,
    price: float = 0.5,
    spread: float = 0.02,
    seed: int = 0,
) -> EpisodeBatch:
    """a corpus containing a feature that is spuriously predictive in-sample.

    the decoy predicts settlement in the FIRST `train_frac` of episodes and is
    pure noise in the remainder. an agent trained on the first part learns to
    condition on it; evaluated on the second part, acting on it earns nothing
    and costs fees.

    this makes a ground-truth-backed test of the two attribution targets:

        per-decision attribution of pi(a|s) on the held-out episodes SHOULD
        credit the decoy, because the policy genuinely does key on it. that is
        a true statement about behaviour and a misleading one about value.

        trajectory-aware attribution of the episode return SHOULD give it
        approximately zero or negative, because on these episodes the decoy
        carries no information and trading on it only pays frictions.

    the correct answer on the held-out part is known by construction, which is
    what makes the disagreement adjudicable rather than merely interesting.

    args:
        decoy_strength: class separation of the decoy during the in-sample part.
        signal_strength: separation of the genuine signal. zero by default, so
            the decoy is the only thing there is to learn.
    """
    rng = np.random.default_rng(seed)

    settlement = rng.integers(0, 2, size=n_episodes).astype(np.float32)
    sign = 2.0 * settlement - 1.0

    n_train = int(n_episodes * train_frac)

    # the genuine signal, usually switched off
    signal = sign[:, None] * signal_strength + rng.normal(
        0.0, noise, size=(n_episodes, n_steps)
    )

    # the decoy: informative in-sample, pure noise out-of-sample
    decoy = rng.normal(0.0, noise, size=(n_episodes, n_steps))
    decoy[:n_train] += sign[:n_train, None] * decoy_strength

    spot = np.zeros((n_episodes, n_steps, N_SPOT), dtype=np.float32)
    spot[:, :, SIGNAL_IDX] = signal.astype(np.float32)
    spot[:, :, DECOY_IDX] = decoy.astype(np.float32)

    half = spread / 2.0
    mid = np.full((n_episodes, n_steps), price, dtype=np.float32)

    return EpisodeBatch(
        bid=(mid - half).astype(np.float32),
        ask=(mid + half).astype(np.float32),
        last_price=mid.copy(),
        volume=np.full((n_episodes, n_steps), 10_000.0, dtype=np.float32),
        staleness=np.zeros((n_episodes, n_steps), dtype=np.float32),
        flow_imbalance=np.zeros((n_episodes, n_steps), dtype=np.float32),
        t_sec=np.tile(
            np.arange(1, n_steps + 1, dtype=np.float32) * 60.0, (n_episodes, 1)
        ),
        settlement=settlement,
        open_epoch=np.arange(n_episodes, dtype=np.float64) * 900.0,
        spot=spot,
    )
