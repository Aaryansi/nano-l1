"""synthetic corpora with known-optimal policies, for agent sanity checks.

the real corpus has no exploitable signal: phase 2 measured the residual
correlation between spot and outcome at +0.016, and the market's own
calibration error at 0.017. that is an honest finding, but it makes the real
data useless for **validating an agent implementation**, because a broken agent
and a correct agent both earn approximately nothing.

so we generate corpora where the answer is known:

    make_learnable_corpus   a genuinely predictive feature exists. a correct
                            agent must find it and earn close to the analytic
                            optimum. if it does not, the agent is broken.

    make_null_corpus        no signal at all, same shapes and frictions. a
                            correct agent must learn to ABSTAIN and earn ~0.
                            if it churns, the agent is broken in the other
                            direction.

both build a real EpisodeBatch and run through the real environment, so these
exercise the actual reward accounting and cost model rather than a mock.
"""

from __future__ import annotations

import numpy as np

from nano_rl.env.binary_market import EpisodeBatch
from nano_rl.env.costs import fee_dollars
from nano_rl.env.features import N_SPOT

# index of spot_ret_since_open within the spot block. the signal is injected
# here because it is the semantically honest place: for a "BTC up?" contract,
# the underlying's move genuinely should predict resolution.
SIGNAL_IDX = 0


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

    because the price never moves, all profit must come from reading the
    signal and holding to settlement. that makes the optimal policy simple and
    its value analytic, which is what we want from a test fixture.

    args:
        signal_strength: separation between the two outcome classes. 1.0 with
            noise 0.3 gives a very learnable but not trivial problem.
        noise: gaussian noise on the signal.
        price: constant mid. 0.5 puts the fee at its maximum, which is the
            adversarial choice and matches the real contract.

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

    the optimal policy here is to stay flat, earning exactly zero. any agent
    that trades is paying frictions for noise.
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

    on `make_learnable_corpus` this IS the optimal policy, because the price is
    constant so there is nothing to gain from trading later, and holding to
    expiry is free. a correct agent should approach this number.

    on `make_null_corpus` it is NOT optimal: the signal is noise, so following
    it just pays frictions. the optimum there is to stay flat and earn exactly
    zero. the function is named for what it computes rather than for what it
    means, because conflating the two is how a benchmark silently becomes
    wrong on half the cases it is used for.

    computed analytically rather than by search, so it cannot inherit a bug
    from the agent it is used to check.
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
