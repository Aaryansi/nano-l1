"""the null-model test with every null matched to its own corpus.

scripts/sanity_check_explanations.py builds ONE null distribution, from agents
trained and attributed on synthetic signal-free corpora, and compares three
observations against it. Two of those observations are themselves on synthetic
corpora built the same way, so the comparison is sound. The third, the real
market agent, is attributed on real Kalshi episodes, and comparing it to spans
measured on synthetic ones puts corpus-to-corpus variation into the reference
distribution that the observation does not have. scripts/null_corpus_check.py
measured the size of that mistake.

Section 3 of the paper defines the null as corrupting the observation channel
while leaving dynamics and reward intact. That is a per-corpus construction: the
null for a case is agents trained on THAT case's episodes with observations
replaced by moment-matched noise, and attributed on the same rollout as the
observation. Nothing else changes. This script implements that definition
literally, for all three cases, so the corpus is held fixed within each
comparison and information is the only thing that varies.

It reports the span directly rather than summing a Shapley attribution. The two
are equal by efficiency, and the span costs two rollouts against a few hundred,
so this is the same statistic computed the cheap way.

The old construction is kept and still run, because the paper reports what was
tried before as well as what it settled on.

usage:
    python scripts/matched_null_test.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus  # noqa: E402
from nano_rl.explain.rollout import (  # noqa: E402
    VectorizedRollout,
    build_background,
    greedy_policy,
    masked_span,
)
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402

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
          f"{'INFORMATIVE' if r.passes else 'not distinguishable'}")

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

    banner("EVERY NULL MATCHED TO ITS OWN CORPUS")
    print("  each case's reference distribution is built by blinding the same")
    print("  episodes the observation is measured on. the corpus is held fixed")
    print("  within a comparison; only the information varies.")

    cases = []

    # 1. planted signal: the test must fire, or it has no power
    lb = make_learnable_corpus(n_episodes=args.episodes, seed=args.seed)
    ln = fit_normalizer(lb)
    la = train_on(BinaryMarketEnv(lb, normalizer=ln, max_position=100.0),
                  args.updates, args.seed)
    cases.append(run_case("planted signal", lb, lb, ln, la, args))

    # 2. a signal-free corpus: the test must decline, or it has no specificity
    hb = make_null_corpus(n_episodes=args.episodes, seed=9999)
    hn = fit_normalizer(hb)
    ha = train_on(BinaryMarketEnv(hb, normalizer=hn, max_position=100.0),
                  args.updates, args.seed + 77)
    cases.append(run_case("null corpus", hb, hb, hn, ha, args))

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
              f"{('informative' if c['fires'] else 'not distinguishable'):>22}")

    by = {c["case"]: c for c in cases}
    power = by["planted signal"]["fires"]
    spec = not by["null corpus"]["fires"]
    print()
    print(f"  fires on a planted signal (power)        : {power}")
    print(f"  declines on a signal-free corpus (spec.) : {spec}")
    if not (power and spec):
        print("\n  the matched construction has lost power or specificity, and")
        print("  the paper cannot adopt it without saying so.")
    else:
        print("\n  the matched construction keeps both. the real-market verdict")
        print(f"  under it is: {'informative' if by['real market']['fires'] else 'not distinguishable'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_null_test.json").write_text(json.dumps({
        "cases": cases,
        "has_power": bool(power),
        "has_specificity": bool(spec),
    }, indent=2))
    print(f"\nwrote {out}/matched_null_test.json")


if __name__ == "__main__":
    main()
