"""the third null construction: permute the outcomes, keep the agent sighted.

the paper has now measured two null constructions and rejected both:

  signal-free corpus   varies the corpus as well as the information, which
                       widens the reference and biases toward declining
                       (scripts/null_corpus_check.py).
  blinded channel      holds the corpus fixed but removes the agent's capacity
                       to respond to observational structure, so the null
                       collapses to a point mass and fires on anything
                       (scripts/matched_null_test.py).

this is the construction the analysis points at and the paper's limitations
section names as missing: hold the corpus fixed AND keep the null agents
sighted, by permuting the outcomes across episodes and leaving everything else
alone. it is the direct analogue of the label-permutation test the paper claims
to be supplying.

the same three cases as matched_null_test, so the results are comparable row for
row. the construction is only worth adopting if it keeps power on a planted
signal AND specificity on a corpus with no signal, which is exactly the pair the
blinded form failed.

usage:
    python scripts/permuted_null_test.py --corpus data/corpus/corpus_candles_60s_spot.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.permuted import outcome_rate, permute_outcomes  # noqa: E402
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus, synthetic_train_eval  # noqa: E402
from nano_rl.explain.rollout import (  # noqa: E402
    VectorizedRollout,
    build_background,
    greedy_policy,
    masked_span,
)
from nano_rl.explain.sanity import (  # noqa: E402
    check_resolving_power,
    UNRESOLVED,
    test_span_against_null,
)


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def train_on(env, updates: int, seed: int) -> PPOAgent:
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def verdict_of(r) -> str:
    """a two-sided rejection below the null is not evidence of information.

    test_span_against_null is two-sided, so it rejects when the observed span is
    unusually SMALL as well as unusually large. only the upper tail means the
    observations carried something. an earlier version of this script printed
    "informative" for a span 2.8 sd BELOW its null, which is close to the
    opposite of what that means.
    """
    if r.verdict == UNRESOLVED:
        return "unresolved"
    if not r.passes:
        return "not distinguishable"
    return "informative" if r.z_score > 0 else "below null"


def run_case(name: str, eval_batch, norm, agent, make_null_world, args) -> dict:
    banner(name.upper())
    roll = VectorizedRollout(eval_batch, normalizer=norm, max_position=100.0)

    # the background depends only on observations, and permutation leaves every
    # observation bit identical, so one background is correct for every world
    # here and rebuilding it per draw would produce the same array.
    bg = build_background(roll, n_samples=192, seed=args.seed)

    observed = masked_span(agent, roll, bg, args.seed)
    ret = float(roll.run(greedy_policy(agent))["returns"].mean())
    print(f"  agent return {ret:+.3f}   span {observed:+.3f}")

    spans, rets = [], []
    for k in range(args.n_null):
        # a permutation test recomputes the WHOLE statistic under the null, so
        # the null agent is trained and measured inside its own permuted world.
        # training there and measuring on the real corpus leaves it out of
        # distribution at measurement time and inflates its span.
        ntrain, neval = make_null_world(k)
        a = train_on(BinaryMarketEnv(ntrain, normalizer=norm,
                                     max_position=100.0), args.updates,
                     args.seed + k)
        nroll = VectorizedRollout(neval, normalizer=norm, max_position=100.0)
        spans.append(masked_span(a, nroll, bg, args.seed + k))
        rets.append(float(nroll.run(greedy_policy(a))["returns"].mean()))
        print(f"    null {k + 1}/{args.n_null}: span {spans[-1]:>+8.3f}",
              flush=True)

    arr = np.array(spans)
    r = test_span_against_null(observed, spans)
    print(f"\n  null {arr.mean():+.3f} +/- {arr.std(ddof=1):.3f}   "
          f"observed {observed:+.3f}   z {r.z_score:+.2f}   "
          f"{verdict_of(r).upper()}")
    print(f"  null agents' mean return {np.mean(rets):+.3f} "
          f"(a collapsed null would sit at 0.000 with sd 0.000)")

    return {"case": name, "return": ret, "span": observed,
            "null_spans": list(map(float, spans)),
            "null_mean": float(arr.mean()), "null_std": float(arr.std(ddof=1)),
            "null_return_mean": float(np.mean(rets)),
            "result": r.as_dict(), "verdict": verdict_of(r),
            "fires": bool(r.passes and r.z_score > 0)}


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
    check_resolving_power(args.n_null, what="the permuted null test null")

    banner("OUTCOME-PERMUTATION NULL")
    print("  observations, dynamics and frictions are untouched. only the")
    print("  settlement each episode is graded against is relabelled, so the")
    print("  null agents stay sighted and the corpus stays fixed.")

    cases = []

    signal_pair = synthetic_train_eval(make_learnable_corpus, n_episodes=args.episodes, seed=args.seed)
    lb = signal_pair.train
    ln = fit_normalizer(lb)
    la = train_on(BinaryMarketEnv(lb, normalizer=ln, max_position=100.0),
                  args.updates, args.seed)
    cases.append(run_case(
        "planted signal", signal_pair.evaluation, ln, la,
        lambda k: (permute_outcomes(lb, 7000 + k),
                   permute_outcomes(signal_pair.evaluation, 17000 + k)), args))

    null_pair = synthetic_train_eval(make_null_corpus, n_episodes=args.episodes, seed=9999)
    hb = null_pair.train
    hn = fit_normalizer(hb)
    ha = train_on(BinaryMarketEnv(hb, normalizer=hn, max_position=100.0),
                  args.updates, args.seed + 77)
    cases.append(run_case(
        "null corpus", null_pair.evaluation, hn, ha,
        lambda k: (permute_outcomes(hb, 7000 + k),
                   permute_outcomes(null_pair.evaluation, 17000 + k)), args))

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.runs}; train first")
    ra = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))
    # train and test are relabelled independently: under the null the outcomes
    # are exchangeable within each split, and permuting the concatenation would
    # move outcomes across the walk-forward boundary for no benefit.
    cases.append(run_case(
        "real market", split.test, split.normalizer, ra,
        lambda k, sp=split: (permute_outcomes(sp.train, 7000 + k),
                             permute_outcomes(sp.test, 8000 + k)), args))

    banner("VERDICT")
    print(f"  {'case':<18}{'span':>10}{'null mean':>12}{'null sd':>10}"
          f"{'z':>9}{'verdict':>22}")
    for c in cases:
        print(f"  {c['case']:<18}{c['span']:>+10.2f}{c['null_mean']:>+12.2f}"
              f"{c['null_std']:>10.2f}{c['result']['z_score']:>+9.2f}"
              f"{c['verdict']:>22}")

    print("\n  These are single control trials, not estimates of power or specificity.")
    collapsed = any(c["null_std"] < 1e-6 for c in cases)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "permuted_null_test.json").write_text(json.dumps({
        "cases": cases, "any_collapsed": bool(collapsed),
        "synthetic_data": {"signal": signal_pair.metadata(), "null": null_pair.metadata()},
    }, indent=2))
    print(f"\nwrote {out}/permuted_null_test.json")


if __name__ == "__main__":
    main()
