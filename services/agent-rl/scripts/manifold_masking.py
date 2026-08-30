"""does the null test measure information, or distribution shift?

the standard objection to perturbation attribution: v(S) replaces the features
outside S with draws that ignore the features inside S, so the policy is scored
on states that never occur. slack et al. (2020) weaponise exactly that. if the
span is an artefact of off-manifold evaluation then the headline result is
about the masking, not about the agent.

this script answers it in four parts.

1. the span is masking-mode invariant BY CONSTRUCTION, because v(N) masks
   nothing and v(empty) replaces the whole observation with a single real
   background row. neither is off-manifold, and with an empty kept set there is
   nothing to condition on. the two modes must agree to sampling noise. this is
   a falsifiable prediction about our own implementation, so we check it.

2. marginal masking really is off-manifold at intermediate coalitions. we
   measure the distance from synthetic states to the nearest real state under
   both modes. if the numbers were the same, the conditional mode would be
   doing nothing and part 3 would be vacuous.

3. the per-feature decomposition is where the objection bites. leave-one-out
   values are recomputed under conditional masking and compared by rank.

4. the verdict itself is re-derived under conditional masking against a
   conditionally-masked null, on both a planted signal and the real market.

usage:
    python scripts/manifold_masking.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    fit_normalizer,
)
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus  # noqa: E402
from nano_rl.explain.manifold import (  # noqa: E402
    masked_value_fn,
    offmanifold_distance,
)
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402

FULL = np.ones(N_FEATURES, dtype=bool)
EMPTY = np.zeros(N_FEATURES, dtype=bool)


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train(batch, norm, updates, seed) -> PPOAgent:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def span_and_loo(agent, batch, bg, norm, n_ep, seed, mode, k, with_loo=True):
    """span, and optionally the leave-one-out vector, under one masking mode."""
    v = masked_value_fn(agent, batch, bg, norm, 100.0, n_ep, seed, mode=mode, k=k)
    v_full = v(FULL)
    span = v_full - v(EMPTY)
    if not with_loo:
        return span, None
    loo = np.empty(N_FEATURES)
    for i in range(N_FEATURES):
        m = FULL.copy()
        m[i] = False
        loo[i] = v_full - v(m)
    return span, loo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"k_neighbours": args.k, "n_null": args.n_null}

    # ------------------------------------------------------- the real agent
    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.runs}; train first")
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))
    rroll = VectorizedRollout(split.test, normalizer=split.normalizer,
                              max_position=100.0)
    rbg = build_background(rroll, n_samples=192, seed=args.seed)

    # ------------------------------------------------------------- part 1
    banner("1. IS THE SPAN INVARIANT TO THE MASKING MODE?")
    print("  predicted: yes. v(N) masks nothing and v(empty) substitutes one")
    print("  whole real background row, so neither coalition leaves the")
    print("  manifold and there is nothing to condition on at the empty set.")
    print()
    span_marg, loo_marg = span_and_loo(
        agent, split.test, rbg, split.normalizer, args.episodes, args.seed,
        "marginal", args.k)
    span_cond, loo_cond = span_and_loo(
        agent, split.test, rbg, split.normalizer, args.episodes, args.seed,
        "conditional", args.k)
    print(f"  span, marginal masking    : {span_marg:>+9.3f}")
    print(f"  span, conditional masking : {span_cond:>+9.3f}")
    print(f"  difference                : {span_cond - span_marg:>+9.3f}")
    res["span_marginal"] = span_marg
    res["span_conditional"] = span_cond

    # ------------------------------------------------------------- part 2
    banner("2. IS MARGINAL MASKING ACTUALLY OFF-MANIFOLD?")
    print("  mean distance from a synthetic state to the nearest real state,")
    print("  in per-feature standard deviations. if these agree, the")
    print("  conditional mode is a no-op and part 3 proves nothing.")
    print()
    rng = np.random.default_rng(args.seed)
    obs = rroll.observations(rroll.n_steps // 2,
                             np.zeros(rroll.n_episodes),
                             np.zeros(rroll.n_episodes),
                             np.zeros(rroll.n_episodes))
    print(f"  {'kept features':>16}{'marginal':>12}{'conditional':>14}")
    dist_rows = []
    # both endpoints are included on purpose: they are the two coalitions the
    # span is built from, and the claim is that neither leaves the manifold.
    for n_kept in (N_FEATURES, N_FEATURES - 1, 12, 9, 4, 0):
        m = EMPTY.copy()
        m[:n_kept] = True
        dm = offmanifold_distance(obs, m, rbg, rng, "marginal", args.k)
        dc = offmanifold_distance(obs, m, rbg, rng, "conditional", args.k)
        dist_rows.append({"n_kept": int(n_kept), "marginal": dm,
                          "conditional": dc})
        print(f"  {n_kept:>16}{dm:>12.4f}{dc:>14.4f}")
    res["offmanifold_distance"] = dist_rows

    # ------------------------------------------------------------- part 3
    banner("3. DOES THE PER-FEATURE RANKING MOVE?")
    rank_m = np.argsort(np.argsort(-np.abs(loo_marg)))
    rank_c = np.argsort(np.argsort(-np.abs(loo_cond)))
    corr = float(np.corrcoef(rank_m, rank_c)[0, 1])
    top1 = bool(np.argmax(np.abs(loo_marg)) == np.argmax(np.abs(loo_cond)))
    print(f"  leave-one-out rank correlation, marginal vs conditional: {corr:+.3f}")
    print(f"  same top feature under both modes                      : {top1}")
    res["loo_rank_corr"] = corr
    res["loo_top1_agree"] = top1
    res["loo_marginal"] = list(map(float, loo_marg))
    res["loo_conditional"] = list(map(float, loo_cond))

    # ------------------------------------------------------------- part 4
    banner(f"4. THE VERDICT UNDER CONDITIONAL MASKING, {args.n_null} nulls")
    nulls = []
    for i in range(args.n_null):
        nb = make_null_corpus(n_episodes=900, seed=1000 + i)
        nn = fit_normalizer(nb)
        na = train(nb, nn, args.updates, args.seed + i)
        nroll = VectorizedRollout(nb, normalizer=nn, max_position=100.0)
        nbg = build_background(nroll, n_samples=192, seed=args.seed + i)
        s, _ = span_and_loo(na, nb, nbg, nn, args.episodes, args.seed + i,
                            "conditional", args.k, with_loo=False)
        nulls.append(s)
        print(f"  null {i + 1}/{args.n_null}: span {s:>+8.2f}", flush=True)

    lb = make_learnable_corpus(n_episodes=900, seed=args.seed)
    ln = fit_normalizer(lb)
    la = train(lb, ln, args.updates, args.seed)
    lroll = VectorizedRollout(lb, normalizer=ln, max_position=100.0)
    lbg = build_background(lroll, n_samples=192, seed=args.seed)
    span_signal, _ = span_and_loo(la, lb, lbg, ln, args.episodes, args.seed,
                                  "conditional", args.k, with_loo=False)

    r_signal = test_span_against_null(span_signal, nulls)
    r_real = test_span_against_null(span_cond, nulls)
    print()
    print(f"  {'case':<18}{'span':>10}{'z':>10}{'verdict':>26}")
    for name, r in (("planted signal", r_signal), ("real market", r_real)):
        print(f"  {name:<18}{r.statistic:>+10.2f}{r.z_score:>+10.2f}"
              f"{('informative' if r.passes else 'not distinguishable'):>26}")
    res["conditional_null_spans"] = list(map(float, nulls))
    res["conditional_planted_signal"] = r_signal.as_dict()
    res["conditional_real_market"] = r_real.as_dict()

    # ------------------------------------------------------------ verdict
    banner("VERDICT")
    invariant = abs(span_cond - span_marg) < max(1.0, 0.15 * abs(span_marg))
    holds = r_signal.passes and not r_real.passes
    separated = all(d["marginal"] > d["conditional"] for d in dist_rows[1:-1])
    print(f"  span invariant to masking mode                : {invariant}")
    print(f"  conditional masking is nearer the manifold     : {separated}")
    print(f"  verdict unchanged under conditional masking    : {holds}")
    if invariant and holds:
        print("\n  the headline result is not an artefact of off-manifold")
        print("  masking. the per-feature ranking is the part that moves.")
    else:
        print("\n  the result DOES depend on the masking mode, and the paper")
        print("  must say so.")
    res["span_invariant"] = bool(invariant)
    res["verdict_holds"] = bool(holds)
    res["conditional_nearer_manifold"] = bool(separated)
    res["feature_names"] = list(FEATURE_NAMES)

    (out / "manifold_masking.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/manifold_masking.json")


if __name__ == "__main__":
    main()
