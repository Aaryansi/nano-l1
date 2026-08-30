"""does the matched null's width depend on how long its agents train?

a concern the matched construction raised. on the synthetic corpora every blind
agent converges to exact abstention, so the null collapses to a point mass at
zero and any nonzero span fires. on the real corpus at the same budget the blind
agents still vary, the null has width around 2.2, and the real market's span of
+7.4 sits inside it.

that difference could be about the corpora, or it could be about convergence.
the real training split is roughly four times the size of the synthetic ones, so
the same number of updates is a quarter as many passes over the data. if it is
convergence, then the real-market verdict is a statement about our training
budget rather than about the agent, and the paper cannot report it as the
latter.

this sweeps the blind agents' budget with everything else held fixed: same
observed agent, same measurement corpus, same background, same seeds. if the
null width is flat in the budget, the verdict is a property of the task. if it
shrinks toward zero, the margin is a property of how long we trained, and the
paper has to say so.

usage:
    python scripts/null_budget_check.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.prediction import BlindEnv, observation_moments  # noqa: E402
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
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def train_on(env, updates, seed):
    env.reset(seed=seed)
    a = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    a.train(env, n_updates=updates, verbose=False)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 80, 160])
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
    observed = masked_span(agent, roll, bg, args.seed)

    banner("DOES THE BLINDED-REAL NULL DEPEND ON ITS TRAINING BUDGET?")
    print(f"  observed span {observed:+.3f}, fixed throughout")
    print(f"  train split {len(split.train)} episodes\n")

    mean_o, sd_o = observation_moments(
        BinaryMarketEnv(split.train, normalizer=norm, max_position=100.0),
        seed=args.seed)

    rows = []
    print(f"  {'updates':>9}{'null mean':>12}{'null sd':>10}{'z':>9}"
          f"{'blind return':>14}{'verdict':>22}")
    for updates in args.budgets:
        spans, rets = [], []
        for k in range(args.n_null):
            blind = BlindEnv(
                BinaryMarketEnv(split.train, normalizer=norm, max_position=100.0),
                mean_o, sd_o, seed=5000 + k)
            a = train_on(blind, updates, args.seed + k)
            spans.append(span(a, roll, bg, args.seed + k))
            rets.append(float(roll.run(greedy_policy(a))["returns"].mean()))
        arr = np.array(spans)
        r = test_span_against_null(observed, spans)
        rows.append({"updates": updates, "null_mean": float(arr.mean()),
                     "null_std": float(arr.std(ddof=1)),
                     "blind_return_mean": float(np.mean(rets)),
                     "spans": list(map(float, spans)),
                     "result": r.as_dict(), "fires": bool(r.passes)})
        print(f"  {updates:>9}{arr.mean():>+12.3f}{arr.std(ddof=1):>10.3f}"
              f"{r.z_score:>+9.2f}{np.mean(rets):>+14.3f}"
              f"{('informative' if r.passes else 'not distinguishable'):>22}",
              flush=True)

    banner("VERDICT")
    sds = [r["null_std"] for r in rows]
    verdicts = {r["fires"] for r in rows}
    shrinking = sds[-1] < 0.5 * sds[0]
    print(f"  null sd across budgets: {' -> '.join(f'{v:.2f}' for v in sds)}")
    if len(verdicts) > 1:
        print("\n  THE VERDICT CHANGES WITH THE BUDGET. the real-market result")
        print("  is a statement about how long the null agents trained, and the")
        print("  paper must report it as budget-dependent.")
    elif shrinking:
        print("\n  the verdict is stable, but the null is still tightening, so")
        print("  the margin is budget-dependent even though the sign is not.")
    else:
        print("\n  the null width is flat in the budget. the verdict is a")
        print("  property of the task, not of how long we trained.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "null_budget_check.json").write_text(json.dumps({
        "observed_span": observed, "rows": rows,
        "verdict_stable": len(verdicts) == 1,
        "null_still_shrinking": bool(shrinking),
    }, indent=2))
    print(f"\nwrote {out}/null_budget_check.json")


if __name__ == "__main__":
    main()
