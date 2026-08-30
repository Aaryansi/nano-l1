"""a positive control on real data: same features, same episodes, two objectives.

the objection this answers: every case where the null test fires elsewhere in
this project is synthetic, so perhaps the test detects planted signal rather
than information, and would decline on any real corpus.

the control holds the corpus, the features, the normalizer, the walk-forward
split and the null construction fixed, and varies only the objective.

    prediction   +1 per step the call agrees with settlement. the market's
                 implied probability is calibrated, so this is learnable.
    trading      change in equity net of the exchange's fees. the same price
                 offers nothing to trade against once costs are paid.

if the test fires on the first and declines on the second, it is separating
tasks by whether their observations carry usable information, not by whether
the data is real. that is the claim.

both nulls are built the same way and on the SAME real episodes: agents trained
where the observation channel has been replaced by draws matched to the real
observation moments. this is a stricter comparison than the synthetic null
corpora used elsewhere, because the corpus is held fixed too.

usage:
    python scripts/positive_control.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.features import N_FEATURES  # noqa: E402
from nano_rl.env.prediction import (  # noqa: E402
    BlindEnv,
    PredictionEnv,
    PredictionRollout,
    observation_moments,
)
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
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train(env, updates: int, seed: int) -> PPOAgent:
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def run_task(name: str, make_env, make_roll, split, args, agent=None) -> dict:
    """train (or reuse) an agent, measure the span, build a matched blind null.

    the trading arm passes the paper's own checkpoint rather than training a
    fresh one. a freshly trained agent at this budget is a DIFFERENT agent from
    the one the rest of the paper explains, and it uses its observations more,
    so training one here would have the control and the paper disagreeing about
    the same row for a reason that has nothing to do with the objective.
    """
    banner(f"{name.upper()}: same episodes, same features")

    if agent is None:
        agent = train(make_env(split.train), args.updates, args.seed)
    roll = make_roll(split.test)
    bg = build_background(
        VectorizedRollout(split.test, normalizer=split.normalizer,
                          max_position=100.0),
        n_samples=192, seed=args.seed,
    )
    observed = masked_span(agent, roll, bg, args.seed)
    ret = float(roll.run(greedy_policy(agent))["returns"].mean())
    print(f"  trained agent: return {ret:+.3f}   span {observed:+.3f}")

    # the null: agents trained on these same real episodes with the
    # observation channel carrying nothing.
    moments_env = make_env(split.train)
    mean, sd = observation_moments(moments_env, seed=args.seed)
    nulls = []
    for k in range(args.n_null):
        blind = BlindEnv(make_env(split.train), mean, sd, seed=3000 + k)
        nulls.append(masked_span(train(blind, args.updates, args.seed + k), roll, bg,
                          args.seed + k))
        print(f"  null {k + 1}/{args.n_null}: span {nulls[-1]:>+8.3f}", flush=True)

    r = test_span_against_null(observed, nulls)
    a = np.array(nulls)
    print(f"\n  null      {a.mean():+.3f} +/- {a.std(ddof=1):.3f}")
    print(f"  observed  {observed:+.3f}   z {r.z_score:+.2f}   "
          f"{'INFORMATIVE' if r.passes else 'not distinguishable'}")

    return {
        "task": name, "return": ret, "span": observed,
        "null_spans": list(map(float, nulls)),
        "null_mean": float(a.mean()), "null_std": float(a.std(ddof=1)),
        "result": r.as_dict(), "fires": bool(r.passes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    norm = split.normalizer

    banner("POSITIVE CONTROL ON REAL DATA")
    print("  identical corpus, features, normalizer and split throughout.")
    print("  the objective is the only thing that varies.")
    print(f"  train {len(split.train)} episodes, test {len(split.test)}")

    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.runs}; train first")
    paper_agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))
    print(f"  trading arm reuses {ckpts[0].name}, the checkpoint the paper explains")

    results = [
        run_task(
            "prediction",
            lambda b: PredictionEnv(b, normalizer=norm, max_position=100.0),
            lambda b: PredictionRollout(b, normalizer=norm, max_position=100.0),
            split, args,
        ),
        run_task(
            "trading",
            lambda b: BinaryMarketEnv(b, normalizer=norm, max_position=100.0),
            lambda b: VectorizedRollout(b, normalizer=norm, max_position=100.0),
            split, args, agent=paper_agent,
        ),
    ]

    banner("VERDICT")
    print(f"  {'task':<14}{'return':>10}{'span':>10}{'z':>10}{'verdict':>24}")
    for r in results:
        print(f"  {r['task']:<14}{r['return']:>+10.2f}{r['span']:>+10.2f}"
              f"{r['result']['z_score']:>+10.2f}"
              f"{('informative' if r['fires'] else 'not distinguishable'):>24}")

    pred = next(r for r in results if r["task"] == "prediction")
    trade = next(r for r in results if r["task"] == "trading")
    separated = pred["fires"] and not trade["fires"]
    print()
    if separated:
        print("  the test fires on the prediction task and declines on the")
        print("  trading task, on identical real episodes. it is separating")
        print("  tasks by whether observations carry usable information,")
        print("  not by whether the corpus is synthetic.")
    elif pred["fires"] and trade["fires"]:
        print("  the test fires on BOTH. the trading result elsewhere in this")
        print("  project does not survive a blinded-real null and the paper")
        print("  must say so.")
    else:
        print("  the test does NOT fire on a task that is learnable from a")
        print("  calibrated price. that is a power failure on real data and")
        print("  the paper must say so.")

    (out / "positive_control.json").write_text(json.dumps({
        "tasks": results,
        "separated": bool(separated),
        "n_train": len(split.train),
        "n_test": len(split.test),
    }, indent=2))
    print(f"\nwrote {out}/positive_control.json")


if __name__ == "__main__":
    main()
