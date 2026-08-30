"""which null does the market verdict actually depend on?

the positive control raised a problem. measured against a null of agents
trained on synthetic signal-free CORPORA, the real market agent's span is
unremarkable (z = +0.23). measured against a null of agents trained on the real
episodes with the observation channel blinded, a freshly trained agent looked
informative. those cannot both be the last word, and the difference matters
because one of them is the paper's headline.

two things differ between the comparisons and they have to be separated:

  the agent        the paper explains a checkpoint trained for 100 updates;
                   the control trained a fresh one for 40.

  the null         synthetic-corpus nulls measure their spans on SYNTHETIC
                   episodes, while the observed span is measured on real test
                   episodes. blinded-real nulls measure on the same real
                   episodes as the observation.

this script holds the agent fixed at the paper's checkpoint and the measurement
corpus fixed at the real test split, and varies only the null construction. any
remaining difference is attributable to the null and to nothing else.

usage:
    python scripts/null_corpus_check.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
    greedy_policy,
    masked_span,
)
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402

FULL = np.ones(N_FEATURES, dtype=bool)
EMPTY = np.zeros(N_FEATURES, dtype=bool)


def banner(t):
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train(env, updates, seed):
    env.reset(seed=seed)
    a = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    a.train(env, n_updates=updates, verbose=False)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    norm = split.normalizer

    ckpt = sorted(Path(args.runs).glob("seed*.pt"))[0]
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpt))

    test_roll = VectorizedRollout(split.test, normalizer=norm, max_position=100.0)
    bg = build_background(test_roll, n_samples=192, seed=args.seed)
    observed = masked_span(agent, test_roll, bg, args.seed)
    ret = float(test_roll.run(greedy_policy(agent))["returns"].mean())

    banner("THE PAPER'S AGENT, MEASURED ON THE REAL TEST SPLIT")
    print(f"  checkpoint {ckpt.name}   return {ret:+.3f}   span {observed:+.3f}")

    # ---- null A: synthetic signal-free corpora, spans measured on themselves
    banner(f"NULL A: synthetic corpora ({args.n_null} agents)")
    print("  each agent trains on its own synthetic signal-free corpus and its")
    print("  span is measured there. the corpus differs from the observation's.")
    a_spans = []
    for k in range(args.n_null):
        nb = make_null_corpus(n_episodes=900, seed=1000 + k)
        nn = fit_normalizer(nb)
        ag = train(BinaryMarketEnv(nb, normalizer=nn, max_position=100.0),
                   args.updates, args.seed + k)
        r = VectorizedRollout(nb, normalizer=nn, max_position=100.0)
        a_spans.append(span(ag, r, build_background(r, 192, args.seed + k),
                            args.seed + k))
        print(f"  {k + 1}/{args.n_null}: {a_spans[-1]:>+8.3f}", flush=True)

    # ---- null B: real episodes, observation channel blinded, spans on test
    banner(f"NULL B: blinded real episodes ({args.n_null} agents)")
    print("  each agent trains on the REAL train split with observations")
    print("  replaced by moment-matched noise; span measured on the real test")
    print("  split, the same corpus the observation uses.")
    mean, sd = observation_moments(
        BinaryMarketEnv(split.train, normalizer=norm, max_position=100.0),
        seed=args.seed)
    b_spans = []
    for k in range(args.n_null):
        blind = BlindEnv(
            BinaryMarketEnv(split.train, normalizer=norm, max_position=100.0),
            mean, sd, seed=3000 + k)
        b_spans.append(masked_span(train(blind, args.updates, args.seed + k),
                            test_roll, bg, args.seed + k))
        print(f"  {k + 1}/{args.n_null}: {b_spans[-1]:>+8.3f}", flush=True)

    ra = test_span_against_null(observed, a_spans)
    rb = test_span_against_null(observed, b_spans)

    banner("SAME AGENT, SAME MEASUREMENT CORPUS, TWO NULLS")
    print(f"  observed span {observed:+.3f}\n")
    print(f"  {'null':<26}{'mean':>10}{'sd':>10}{'z':>9}{'verdict':>24}")
    for nm, sp, r in (("A synthetic corpora", a_spans, ra),
                      ("B blinded real episodes", b_spans, rb)):
        arr = np.array(sp)
        print(f"  {nm:<26}{arr.mean():>+10.3f}{arr.std(ddof=1):>10.3f}"
              f"{r.z_score:>+9.2f}"
              f"{('informative' if r.passes else 'not distinguishable'):>24}")

    agree = ra.passes == rb.passes
    print()
    if agree:
        print("  the two constructions agree. the paper's verdict does not")
        print("  depend on which null is used.")
    else:
        print("  THE TWO CONSTRUCTIONS DISAGREE. the paper's headline verdict")
        print("  depends on the null construction, and the paper must say so")
        print("  and argue for the one it uses.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "null_corpus_check.json").write_text(json.dumps({
        "checkpoint": ckpt.name, "observed_span": observed, "return": ret,
        "null_synthetic": {"spans": a_spans, "result": ra.as_dict()},
        "null_blinded_real": {"spans": b_spans, "result": rb.as_dict()},
        "constructions_agree": bool(agree),
    }, indent=2))
    print(f"\nwrote {args.out}/null_corpus_check.json")


if __name__ == "__main__":
    main()
