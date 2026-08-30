"""is the null-test verdict specific to how credit is assigned?

the integrated-gradients comparison could not answer this, because episode
return is not differentiable through the environment, so IG has no outcome-level
analogue. this closes it with two schemes that ARE outcome level and
perturbation based, sharing Shapley's masking but not its credit assignment:

    leave-one-out   phi_i = v(N) - v(N \\ {i})
    only-one-in     phi_i = v({i}) - v(empty)

Shapley averages marginal contributions over all coalition sizes; these are the
two extremes of that average. crucially neither satisfies efficiency, so each
has its own total, distinct from the span. that makes them genuine alternative
statistics rather than the same number computed twice.

each is tested against a null built with the same scheme, on:
  a planted signal   the test must fire
  the real market    the question

usage:
    python scripts/scheme_robustness.py --corpus data/corpus/corpus_candles_60s_spot.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl import plots  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import FEATURE_NAMES, N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus  # noqa: E402
from nano_rl.explain.outcome_schemes import (  # noqa: E402
    leave_one_out,
    only_one_in,
    span_only,
)
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train(batch, norm, updates, seed) -> PPOAgent:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def all_schemes(agent, batch, bg, norm, n_ep, seed) -> dict[str, float]:
    """the three totals for one agent."""
    loo_v, loo_t = leave_one_out(agent, batch, bg, norm, n_episodes=n_ep, seed=seed)
    ooi_v, ooi_t = only_one_in(agent, batch, bg, norm, n_episodes=n_ep, seed=seed)
    sp = span_only(agent, batch, bg, norm, n_episodes=n_ep, seed=seed)
    return {"span": sp, "leave_one_out": loo_t, "only_one_in": ooi_t,
            "loo_values": loo_v, "ooi_values": ooi_v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=16)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    banner("0. THE STATISTIC IS NOT A SHAPLEY QUANTITY")
    print("  span = v(all) - v(none) is the difference between two masked")
    print("  rollouts. shapley's efficiency axiom says the shapley values sum")
    print("  to it, which is why it can be read off an attribution, but the")
    print("  span itself depends only on the masking. so part of the question")
    print("  is answered by construction.")
    print()
    print("  what could still be scheme dependent is the per-feature")
    print("  decomposition, and whether a scheme with a DIFFERENT total")
    print("  reaches the same verdict. leave-one-out and only-one-in do not")
    print("  satisfy efficiency, so each carries its own total.")

    # ------------------------------------------------------------- nulls
    banner(f"1. NULL DISTRIBUTIONS, {args.n_null} agents per scheme")
    nulls: dict[str, list[float]] = {"span": [], "leave_one_out": [], "only_one_in": []}
    for k in range(args.n_null):
        nb = make_null_corpus(n_episodes=900, seed=1000 + k)
        nn = fit_normalizer(nb)
        agent = train(nb, nn, args.updates, args.seed + k)
        roll = VectorizedRollout(nb, normalizer=nn, max_position=100.0)
        bg = build_background(roll, n_samples=192, seed=args.seed + k)
        r = all_schemes(agent, nb, bg, nn, args.episodes, args.seed + k)
        for key in nulls:
            nulls[key].append(r[key])
        print(f"  null {k + 1}/{args.n_null}: span {r['span']:>+8.2f}  "
              f"loo {r['leave_one_out']:>+8.2f}  ooi {r['only_one_in']:>+8.2f}",
              flush=True)

    for key in nulls:
        a = np.array(nulls[key])
        print(f"  {key:<16} {a.mean():>+9.3f} +/- {a.std(ddof=1):>7.3f}")

    # -------------------------------------------------------- power check
    banner("2. PLANTED SIGNAL: every scheme must fire")
    lb = make_learnable_corpus(n_episodes=900, seed=args.seed)
    ln = fit_normalizer(lb)
    la = train(lb, ln, args.updates, args.seed)
    lroll = VectorizedRollout(lb, normalizer=ln, max_position=100.0)
    lbg = build_background(lroll, n_samples=192, seed=args.seed)
    lr = all_schemes(la, lb, lbg, ln, args.episodes, args.seed)

    signal_res = {}
    print(f"  {'scheme':<18}{'statistic':>12}{'z':>10}{'verdict':>26}")
    for key in nulls:
        r = test_span_against_null(lr[key], nulls[key])
        signal_res[key] = r
        print(f"  {key:<18}{r.statistic:>+12.2f}{r.z_score:>+10.2f}"
              f"{('informative' if r.passes else 'not distinguishable'):>26}")

    # ----------------------------------------------------- the real agent
    banner("3. THE REAL MARKET: does the verdict depend on the scheme?")
    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))

    rroll = VectorizedRollout(split.test, normalizer=split.normalizer,
                              max_position=100.0)
    rbg = build_background(rroll, n_samples=192, seed=args.seed)
    rr = all_schemes(agent, split.test, rbg, split.normalizer,
                     args.episodes, args.seed)

    real_res = {}
    print(f"  {'scheme':<18}{'statistic':>12}{'z':>10}{'verdict':>26}")
    for key in nulls:
        r = test_span_against_null(rr[key], nulls[key])
        real_res[key] = r
        print(f"  {key:<18}{r.statistic:>+12.2f}{r.z_score:>+10.2f}"
              f"{('informative' if r.passes else 'not distinguishable'):>26}")

    # ------------------------------------------------------------ verdict
    banner("VERDICT")
    all_fire = all(signal_res[k].passes for k in nulls)
    none_fire = not any(real_res[k].passes for k in nulls)
    print(f"  every scheme fires on the planted signal : {all_fire}")
    print(f"  no scheme fires on the real market       : {none_fire}")
    if all_fire and none_fire:
        print("\n  the verdict does not depend on how credit is assigned.")
        print("  the finding is a property of the agent, not of shapley.")
    else:
        print("\n  the schemes DISAGREE, so the verdict is scheme dependent")
        print("  and the paper must say so.")

    # per-feature agreement between the two non-efficient schemes
    corr = float(np.corrcoef(
        np.argsort(np.argsort(-np.abs(rr["loo_values"]))),
        np.argsort(np.argsort(-np.abs(rr["ooi_values"]))),
    )[0, 1])
    print(f"\n  leave-one-out vs only-one-in, per-feature rank correlation: "
          f"{corr:+.3f}")

    plots.scheme_comparison(
        {k: (np.array(nulls[k]), lr[k], rr[k]) for k in nulls},
        out / "scheme_robustness.png",
        subtitle=f"{args.n_null} null agents per scheme; the planted signal must "
                 f"fire and the real market must not",
    )

    (out / "scheme_robustness.json").write_text(json.dumps({
        "nulls": {k: list(map(float, v)) for k, v in nulls.items()},
        "planted_signal": {k: signal_res[k].as_dict() for k in nulls},
        "real_market": {k: real_res[k].as_dict() for k in nulls},
        "all_schemes_fire_on_signal": bool(all_fire),
        "no_scheme_fires_on_real": bool(none_fire),
        "loo_vs_ooi_rank_corr": corr,
        "feature_names": list(FEATURE_NAMES),
    }, indent=2))
    print(f"\nwrote {out}/scheme_robustness.json")


if __name__ == "__main__":
    main()
