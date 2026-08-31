"""does the null-construction problem hold across all three SVERL targets?

everything this paper reports about null construction is measured on the
OUTCOME-level statistic: the span of expected episode return. the obvious
objection is that the failures might be a property of that particular
characteristic function rather than of explaining reinforcement learning agents.

beechey, smith and simsek (SVERL, ICML 2023) identify three explanatory targets,
and they are genuinely different questions about the same agent:

    behaviour    what did the observation do to pi(a* | s), the action taken
    prediction   what did it do to V(s), the critic's estimate
    outcomes     what did it do to the return actually collected

we already implement all three. this measures the span under each and asks two
questions of each:

  1. do the three targets agree on the real market agent, against the same null?
  2. does the blinded construction degenerate for all three, or only for
     outcomes?

question 2 is the one that matters. if blinding collapses the reference to a
point mass for every target, the failure is a property of the CONSTRUCTION and
generalises past our choice of statistic. if it collapses only for outcomes, our
result is narrower than the paper currently claims and the paper must say so.

the span is read off the two endpoint coalitions for every target, which is
exact rather than estimated: by efficiency the shapley values sum to it, and
evaluating v(N) and v(empty) directly avoids inheriting permutation-sampling
noise into a quantity that does not need it.

usage:
    python scripts/sverl_targets.py --corpus data/corpus/corpus_candles_60s_spot.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.prediction import BlindEnv, observation_moments  # noqa: E402
from nano_rl.env.synthetic import make_null_corpus  # noqa: E402
from nano_rl.explain.rollout import (  # noqa: E402
    VectorizedRollout,
    build_background,
    masked_span,
)
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402

TARGETS = ("behaviour", "prediction", "outcomes")


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def _mask_all(obs: np.ndarray, bg: np.ndarray, rng, n_draws: int) -> np.ndarray:
    """n_draws fully-masked versions of each row of obs.

    the empty coalition replaces every feature, so the synthetic state is one
    whole background row. drawing several and averaging is what makes v(empty)
    an expectation rather than one sample.
    """
    idx = rng.integers(0, len(bg), size=(n_draws, len(obs)))
    return bg[idx]


def span_behaviour(agent, obs, bg, seed, n_draws=64) -> float:
    """v(N) - v(empty) for pi(a* | s), averaged over states.

    a* is the action the agent takes on the REAL observation, held fixed while
    the observation is masked. that is the SVERL behaviour question: what did
    seeing the state do to the probability of the thing it actually did.
    """
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        logits, _ = agent.net(torch.as_tensor(obs, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).numpy()
        a_star = probs.argmax(axis=-1)
        full = probs[np.arange(len(obs)), a_star]

        masked = _mask_all(obs, bg, rng, n_draws)
        flat = masked.reshape(-1, obs.shape[1])
        ml, _ = agent.net(torch.as_tensor(flat, dtype=torch.float32))
        mp = torch.softmax(ml, dim=-1).numpy().reshape(n_draws, len(obs), -1)
        empty = mp[:, np.arange(len(obs)), a_star].mean(axis=0)
    return float((full - empty).mean())


def span_prediction(agent, obs, bg, seed, n_draws=64) -> float:
    """v(N) - v(empty) for the critic's V(s), averaged over states."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        _, v_full = agent.net(torch.as_tensor(obs, dtype=torch.float32))
        masked = _mask_all(obs, bg, rng, n_draws)
        flat = masked.reshape(-1, obs.shape[1])
        _, v_masked = agent.net(torch.as_tensor(flat, dtype=torch.float32))
        v_masked = v_masked.numpy().reshape(n_draws, len(obs)).mean(axis=0)
    return float((v_full.numpy().ravel() - v_masked).mean())


def all_spans(agent, roll, obs, bg, seed) -> dict[str, float]:
    return {
        "behaviour": span_behaviour(agent, obs, bg, seed),
        "prediction": span_prediction(agent, obs, bg, seed),
        "outcomes": masked_span(agent, roll, bg, seed),
    }


def states_of(roll: VectorizedRollout, n: int, seed: int) -> np.ndarray:
    """a sample of real observations, taken with the agent flat."""
    rng = np.random.default_rng(seed)
    z = np.zeros(roll.n_episodes)
    rows = [roll.observations(t, z, z, z) for t in range(roll.n_steps)]
    stacked = np.concatenate(rows, axis=0)
    idx = rng.choice(len(stacked), size=min(n, len(stacked)), replace=False)
    return stacked[idx]


def train_on(env, updates: int, seed: int) -> PPOAgent:
    env.reset(seed=seed)
    a = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    a.train(env, n_updates=updates, verbose=False)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=16)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--states", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    norm = split.normalizer
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.runs}; train first")
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))

    roll = VectorizedRollout(split.test, normalizer=norm, max_position=100.0)
    bg = build_background(roll, n_samples=192, seed=args.seed)
    obs = states_of(roll, args.states, args.seed)

    banner("THE THREE SVERL TARGETS ON THE REAL MARKET AGENT")
    observed = all_spans(agent, roll, obs, bg, args.seed)
    for t in TARGETS:
        print(f"  {t:<12} span {observed[t]:+.4f}")

    # ---- null A: the construction the paper uses
    banner(f"NULL A: signal-free corpora ({args.n_null} agents)")
    a_nulls = {t: [] for t in TARGETS}
    for k in range(args.n_null):
        nb = make_null_corpus(n_episodes=900, seed=1000 + k)
        nn = fit_normalizer(nb)
        ag = train_on(BinaryMarketEnv(nb, normalizer=nn, max_position=100.0),
                      args.updates, args.seed + k)
        nr = VectorizedRollout(nb, normalizer=nn, max_position=100.0)
        nbg = build_background(nr, n_samples=192, seed=args.seed + k)
        s = all_spans(ag, nr, states_of(nr, args.states, args.seed + k), nbg,
                      args.seed + k)
        for t in TARGETS:
            a_nulls[t].append(s[t])
        print(f"  {k + 1}/{args.n_null}: " +
              "  ".join(f"{t[:4]} {s[t]:+7.3f}" for t in TARGETS), flush=True)

    # ---- null B: blinding, which collapsed for outcomes
    banner(f"NULL B: blinded real episodes ({args.n_null} agents)")
    mean_o, sd_o = observation_moments(
        BinaryMarketEnv(split.train, normalizer=norm, max_position=100.0),
        seed=args.seed)
    b_nulls = {t: [] for t in TARGETS}
    for k in range(args.n_null):
        blind = BlindEnv(
            BinaryMarketEnv(split.train, normalizer=norm, max_position=100.0),
            mean_o, sd_o, seed=5000 + k)
        ag = train_on(blind, args.updates, args.seed + k)
        s = all_spans(ag, roll, obs, bg, args.seed + k)
        for t in TARGETS:
            b_nulls[t].append(s[t])
        print(f"  {k + 1}/{args.n_null}: " +
              "  ".join(f"{t[:4]} {s[t]:+7.3f}" for t in TARGETS), flush=True)

    banner("VERDICTS")
    print(f"  {'target':<12}{'span':>10}{'null A':>22}{'z':>8}{'':>3}"
          f"{'null B':>22}{'z':>8}")
    results = {}
    for t in TARGETS:
        A, B = np.array(a_nulls[t]), np.array(b_nulls[t])
        ra = test_span_against_null(observed[t], a_nulls[t])
        rb = test_span_against_null(observed[t], b_nulls[t])
        results[t] = {
            "span": observed[t],
            "null_signal_free": {"mean": float(A.mean()), "std": float(A.std(ddof=1)),
                                 "spans": a_nulls[t], "result": ra.as_dict()},
            "null_blinded": {"mean": float(B.mean()), "std": float(B.std(ddof=1)),
                             "spans": b_nulls[t], "result": rb.as_dict()},
            "blinded_degenerate": bool(B.std(ddof=1) < 1e-9),
        }
        print(f"  {t:<12}{observed[t]:>+10.4f}"
              f"{f'{A.mean():+.3f} +/- {A.std(ddof=1):.3f}':>22}{ra.z_score:>+8.2f}"
              f"{'':>3}{f'{B.mean():+.3f} +/- {B.std(ddof=1):.3f}':>22}"
              f"{rb.z_score:>+8.2f}")

    banner("WHAT THIS SETTLES")
    degen = [t for t in TARGETS if results[t]["blinded_degenerate"]]
    verdicts_A = {t: results[t]["null_signal_free"]["result"]["passes"]
                  for t in TARGETS}
    agree_A = len(set(verdicts_A.values())) == 1
    fires = [t for t, v in verdicts_A.items() if v]

    print("  under the paper's null, the three targets say:")
    for t in TARGETS:
        z = results[t]["null_signal_free"]["result"]["z_score"]
        print(f"    {t:<12} z {z:>+7.2f}   "
              f"{'informative' if verdicts_A[t] else 'not distinguishable'}")
    print()
    if agree_A:
        print("  the three agree, so the verdict does not depend on which")
        print("  question about the agent is being asked.")
    else:
        print(f"  THE THREE DISAGREE. {', '.join(fires) or 'none'} fires while the")
        print("  others decline, on one agent, one corpus and one null. so the")
        print("  choice of explanatory target decides the verdict, exactly as")
        print("  the choice of null construction does, and neither is usually")
        print("  reported. that is a second degree of freedom, orthogonal to")
        print("  the first.")

    print()
    print("  on degeneracy, this run establishes less than it may appear.")
    print(f"  blinding degenerated for: {', '.join(degen) if degen else 'no target'}.")
    print("  that is expected and is NOT evidence against the outcome-level")
    print("  result: the collapse reported elsewhere was found on synthetic")
    print("  corpora, and on the real corpus only once the blind agents had")
    print("  converged, near 160 updates. this run uses the real corpus at 40,")
    print("  which is neither condition. it is not a test of that claim.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sverl_targets.json").write_text(json.dumps({
        "targets": results,
        "targets_agree_under_signal_free_null": bool(agree_A),
        "blinded_degenerate_targets": degen,
        "n_null": args.n_null, "n_states": args.states,
    }, indent=2))
    print(f"\nwrote {out}/sverl_targets.json")


if __name__ == "__main__":
    main()
