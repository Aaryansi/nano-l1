"""Compare original-input and noise-trained policies on matched populations.

Each synthetic comparison uses independent training and evaluation draws;
the market uses chronological training and test splits. Noise replacement
changes the policy's training input distribution and ability to respond to
state. The environment and evaluation population are matched within each case,
but this does not isolate a single information mechanism or prove exchangeability.
Endpoint return differences avoid estimating individual Shapley coordinates;
they cannot establish the fidelity of a feature decomposition.
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
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus, synthetic_train_eval  # noqa: E402
from nano_rl.explain.rollout import (  # noqa: E402
    VectorizedRollout,
    build_background,
    greedy_policy,
    masked_span,
)
from nano_rl.explain.sanity import (  # noqa: E402
    check_resolving_power,
    short_verdict,
    test_span_against_null,
)

FULL = np.ones(N_FEATURES, dtype=bool)
EMPTY = np.zeros(N_FEATURES, dtype=bool)


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def train_on(env, updates: int, seed: int) -> PPOAgent:
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def matched_null(train_batch, norm, roll, bg, n_null: int, updates: int,
                 seed: int) -> list[float]:
    """spans of agents trained on THIS corpus with the observations blinded.

    the moments are estimated from the same corpus, so the fake observations
    occupy the range the real ones do. a network fed out-of-range inputs fails
    for reasons of scale rather than of information, which would be a different
    experiment.
    """
    mean, sd = observation_moments(
        BinaryMarketEnv(train_batch, normalizer=norm, max_position=100.0),
        seed=seed)
    spans = []
    for k in range(n_null):
        blind = BlindEnv(
            BinaryMarketEnv(train_batch, normalizer=norm, max_position=100.0),
            mean, sd, seed=5000 + k)
        spans.append(masked_span(train_on(blind, updates, seed + k), roll, bg, seed + k))
        print(f"    null {k + 1}/{n_null}: {spans[-1]:>+8.3f}", flush=True)
    return spans


def run_case(name: str, train_batch, eval_batch, norm, agent, args) -> dict:
    banner(f"{name.upper()}")
    roll = VectorizedRollout(eval_batch, normalizer=norm, max_position=100.0)
    bg = build_background(roll, n_samples=192, seed=args.seed)

    observed = masked_span(agent, roll, bg, args.seed)
    ret = float(roll.run(greedy_policy(agent))["returns"].mean())
    print(f"  agent return {ret:+.3f}   span {observed:+.3f}")
    print(f"  null: {args.n_null} agents trained on these episodes, blinded")

    nulls = matched_null(train_batch, norm, roll, bg, args.n_null,
                         args.updates, args.seed)
    a = np.array(nulls)
    r = test_span_against_null(observed, nulls)
    print(f"\n  null {a.mean():+.3f} +/- {a.std(ddof=1):.3f}   "
          f"observed {observed:+.3f}   z {r.z_score:+.2f}   "
          f"{short_verdict(r)}")

    return {"case": name, "return": ret, "span": observed,
            "null_spans": list(map(float, nulls)),
            "null_mean": float(a.mean()), "null_std": float(a.std(ddof=1)),
            "result": r.as_dict(), "fires": bool(r.passes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=24)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    check_resolving_power(args.n_null, what="the matched null test null")

    banner("EVERY NULL MATCHED TO ITS OWN CORPUS")
    print("  each case's reference distribution is built by blinding the same")
    print("  episodes the observation is measured on. the corpus is held fixed")
    print("  within a comparison; training input distributions also change.")

    cases = []

    # 1. planted signal: the test must fire, or it has no power
    signal_pair = synthetic_train_eval(make_learnable_corpus, n_episodes=args.episodes, seed=args.seed)
    lb = signal_pair.train
    ln = fit_normalizer(lb)
    la = train_on(BinaryMarketEnv(lb, normalizer=ln, max_position=100.0),
                  args.updates, args.seed)
    cases.append(run_case("planted signal", lb, signal_pair.evaluation, ln, la, args))

    # 2. a signal-free corpus: the test must decline, or it has no specificity
    null_pair = synthetic_train_eval(make_null_corpus, n_episodes=args.episodes, seed=9999)
    hb = null_pair.train
    hn = fit_normalizer(hb)
    ha = train_on(BinaryMarketEnv(hb, normalizer=hn, max_position=100.0),
                  args.updates, args.seed + 77)
    cases.append(run_case("null corpus", hb, null_pair.evaluation, hn, ha, args))

    # 3. the real market, trained on the real train split and evaluated once
    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.runs}; train first")
    ra = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))
    cases.append(run_case("real market", split.train, split.test,
                          split.normalizer, ra, args))

    banner("VERDICT")
    print(f"  {'case':<18}{'span':>10}{'null mean':>12}{'null sd':>10}"
          f"{'z':>9}{'verdict':>22}")
    for c in cases:
        print(f"  {c['case']:<18}{c['span']:>+10.2f}{c['null_mean']:>+12.2f}"
              f"{c['null_std']:>10.2f}{c['result']['z_score']:>+9.2f}"
              f"{short_verdict(c['result']):>22}")

    print("\n  These are single control trials, not estimates of power or specificity.")
    collapsed = any(c["null_std"] < 1e-6 for c in cases)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_null_test.json").write_text(json.dumps({
        "cases": cases,
        "synthetic_data": {"signal": signal_pair.metadata(), "null": null_pair.metadata()},
    }, indent=2))
    print(f"\nwrote {out}/matched_null_test.json")


if __name__ == "__main__":
    main()
