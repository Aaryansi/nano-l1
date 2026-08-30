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
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus  # noqa: E402
from nano_rl.explain.rollout import (  # noqa: E402
    VectorizedRollout,
    build_background,
    greedy_policy,
    masked_span,
)
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def train_on(env, updates: int, seed: int) -> PPOAgent:
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def run_case(name: str, train_batch, eval_batch, norm, agent, args) -> dict:
    banner(name.upper())
    roll = VectorizedRollout(eval_batch, normalizer=norm, max_position=100.0)
    bg = build_background(roll, n_samples=192, seed=args.seed)

    observed = masked_span(agent, roll, bg, args.seed)
    ret = float(roll.run(greedy_policy(agent))["returns"].mean())
    print(f"  agent return {ret:+.3f}   span {observed:+.3f}")
    print(f"  outcome rate {outcome_rate(train_batch):.4f}, preserved by "
          f"permutation")

    spans, rets = [], []
    for k in range(args.n_null):
        # a fresh permutation per null agent, so the reference reflects
        # variation in the relabelling as well as in training
        pb = permute_outcomes(train_batch, seed=7000 + k)
        env = BinaryMarketEnv(pb, normalizer=norm, max_position=100.0)
        a = train_on(env, args.updates, args.seed + k)
        spans.append(masked_span(a, roll, bg, args.seed + k))
        rets.append(float(roll.run(greedy_policy(a))["returns"].mean()))
        print(f"    null {k + 1}/{args.n_null}: span {spans[-1]:>+8.3f}",
              flush=True)

    arr = np.array(spans)
    r = test_span_against_null(observed, spans)
    print(f"\n  null {arr.mean():+.3f} +/- {arr.std(ddof=1):.3f}   "
          f"observed {observed:+.3f}   z {r.z_score:+.2f}   "
          f"{'INFORMATIVE' if r.passes else 'not distinguishable'}")
    print(f"  null agents' mean return {np.mean(rets):+.3f} "
          f"(a collapsed null would sit at 0.000 with sd 0.000)")

    return {"case": name, "return": ret, "span": observed,
            "null_spans": list(map(float, spans)),
            "null_mean": float(arr.mean()), "null_std": float(arr.std(ddof=1)),
            "null_return_mean": float(np.mean(rets)),
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

    banner("OUTCOME-PERMUTATION NULL")
    print("  observations, dynamics and frictions are untouched. only the")
    print("  settlement each episode is graded against is relabelled, so the")
    print("  null agents stay sighted and the corpus stays fixed.")

    cases = []

    lb = make_learnable_corpus(n_episodes=args.episodes, seed=args.seed)
    ln = fit_normalizer(lb)
    la = train_on(BinaryMarketEnv(lb, normalizer=ln, max_position=100.0),
                  args.updates, args.seed)
    cases.append(run_case("planted signal", lb, lb, ln, la, args))

    hb = make_null_corpus(n_episodes=args.episodes, seed=9999)
    hn = fit_normalizer(hb)
    ha = train_on(BinaryMarketEnv(hb, normalizer=hn, max_position=100.0),
                  args.updates, args.seed + 77)
    cases.append(run_case("null corpus", hb, hb, hn, ha, args))

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
    collapsed = any(c["null_std"] < 1e-6 for c in cases)
    print()
    print(f"  fires on a planted signal (power)        : {power}")
    print(f"  declines on a signal-free corpus (spec.) : {spec}")
    print(f"  any null collapsed to a point mass       : {collapsed}")
    if power and spec and not collapsed:
        print("\n  this construction keeps both properties the blinded form")
        print("  lost, on the same three cases. it holds the corpus fixed and")
        print(f"  the real-market verdict under it is: "
              f"{'informative' if by['real market']['fires'] else 'not distinguishable'}")
    else:
        print("\n  this construction fails too, and the paper reports three")
        print("  failures rather than a fix.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "permuted_null_test.json").write_text(json.dumps({
        "cases": cases, "has_power": bool(power),
        "has_specificity": bool(spec), "any_collapsed": bool(collapsed),
    }, indent=2))
    print(f"\nwrote {out}/permuted_null_test.json")


if __name__ == "__main__":
    main()
