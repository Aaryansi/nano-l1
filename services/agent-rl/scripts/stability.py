"""do agents that perform identically produce the same explanation?

this asks whether an rl feature attribution is a property of the TASK or a
property of the random seed. it matters practically: if two agents with
statistically indistinguishable performance disagree about which features
matter, then a published explanation of a single trained agent may be an
artifact of initialisation rather than a fact about the problem.

the design controls for the obvious confound. seeds are only compared if their
test performance is statistically indistinguishable, checked by paired
bootstrap. otherwise "different explanations" would just mean "different
agents", which is neither surprising nor interesting.

three quantities per attribution target:

  pairwise rank correlation   do seeds agree on the ORDERING of features?
  top-1 agreement             do they agree on the single most important one?
  sign agreement              where both assign meaningful mass, do they agree
                              on the direction of the effect?

and the comparison that would be a new argument for outcome-based attribution:
is the OUTCOMES target more stable across seeds than the BEHAVIOUR target?

usage:
    python scripts/stability.py --corpus data/corpus/corpus_candles_60s_spot.npz \\
        --runs runs/ppo --out reports
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl import plots  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import FEATURE_NAMES, N_FEATURES  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.shapley import Attribution  # noqa: E402
from nano_rl.explain.trajectory import (  # noqa: E402
    OutcomeAttributionConfig,
    explain_behaviour,
    explain_outcomes,
)
from nano_rl.metrics import paired_bootstrap_p_value  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def global_behaviour_attribution(
    agent: PPOAgent, env: BinaryMarketEnv, background: np.ndarray,
    n_states: int = 30, n_permutations: int = 60, seed: int = 0,
) -> Attribution:
    """mean absolute per-decision attribution over sampled states.

    a single-state attribution is dominated by where that state happens to sit
    relative to the background, so the global summary is what gets compared.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros(N_FEATURES)

    for i in range(n_states):
        ep = int(rng.integers(0, len(env.batch)))
        obs, _ = env.reset(options={"episode": ep})
        for _ in range(int(rng.integers(0, env.n_steps - 1))):
            with torch.no_grad():
                a, _, _ = agent.net.act(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                    deterministic=True,
                )
            obs, _, done, _, _ = env.step(int(a.item()))
            if done:
                break
        att, _ = explain_behaviour(
            agent, obs, background, n_permutations=n_permutations, seed=seed + i
        )
        acc += np.abs(att.values)

    acc /= n_states
    return Attribution(
        values=acc, stderr=np.zeros(N_FEATURES), base_value=0.0,
        full_value=float(acc.sum()), feature_names=FEATURE_NAMES,
    )


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-np.abs(a))).astype(float)
    rb = np.argsort(np.argsort(-np.abs(b))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def summarise(name: str, atts: list[np.ndarray]) -> dict:
    """pairwise agreement statistics across seeds."""
    pairs = list(itertools.combinations(range(len(atts)), 2))
    rhos, top1, signs = [], [], []

    for i, j in pairs:
        a, b = atts[i], atts[j]
        rhos.append(rank_correlation(a, b))
        top1.append(int(np.argmax(np.abs(a))) == int(np.argmax(np.abs(b))))

        # sign agreement only where BOTH assign meaningful mass, since the sign
        # of a negligible attribution is noise
        thresh_a = 0.1 * np.abs(a).max()
        thresh_b = 0.1 * np.abs(b).max()
        both = (np.abs(a) > thresh_a) & (np.abs(b) > thresh_b)
        if both.sum() > 0:
            signs.append(float((np.sign(a[both]) == np.sign(b[both])).mean()))

    return {
        "target": name,
        "n_pairs": len(pairs),
        "rank_corr_mean": float(np.mean(rhos)),
        "rank_corr_min": float(np.min(rhos)),
        "rank_corr_std": float(np.std(rhos)),
        "top1_agreement": float(np.mean(top1)),
        "sign_agreement": float(np.mean(signs)) if signs else float("nan"),
        "per_pair_rho": [round(r, 3) for r in rhos],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-states", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)

    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if len(ckpts) < 2:
        print(f"need at least 2 checkpoints in {args.runs}")
        raise SystemExit(1)

    agents = []
    for c in ckpts:
        a = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(c))
        agents.append(a)

    banner(f"1. ARE THESE AGENTS EVEN COMPARABLE? ({len(agents)} seeds)")
    print("  comparing explanations only makes sense between agents whose")
    print("  performance is statistically indistinguishable. otherwise")
    print("  'different explanations' just means 'different agents'.\n")

    env = BinaryMarketEnv(
        split.test, normalizer=split.normalizer, max_position=100.0,
        random_episode_order=False,
    )
    pnls = [a.evaluate(env, n_episodes=len(split.test))["returns"] for a in agents]

    print(f"  {'seed':>5} {'mean pnl':>10}")
    for i, p in enumerate(pnls):
        print(f"  {i:>5} {p.mean():>+10.4f}")

    pairs = list(itertools.combinations(range(len(agents)), 2))
    ps = [paired_bootstrap_p_value(pnls[i], pnls[j]) for i, j in pairs]
    n_indist = sum(1 for p in ps if p >= 0.05)
    print(f"\n  pairwise paired-bootstrap p-values: "
          f"min {min(ps):.3f}, median {float(np.median(ps)):.3f}")
    print(f"  {n_indist} of {len(pairs)} pairs are statistically "
          f"indistinguishable at p >= 0.05")

    if n_indist < len(pairs) // 2:
        print("\n  WARNING: most seeds differ in performance, so any difference")
        print("  in explanation is confounded. interpret with care.")

    banner("2. DO THEY EXPLAIN THEMSELVES THE SAME WAY?")

    roll = VectorizedRollout(
        split.test, normalizer=split.normalizer, max_position=100.0
    )
    background = build_background(roll, n_samples=256, seed=args.seed)

    behaviour, outcomes = [], []
    for i, agent in enumerate(agents):
        print(f"  seed {i}: attributing...", flush=True)
        behaviour.append(
            global_behaviour_attribution(
                agent, env, background, n_states=args.n_states, seed=args.seed
            ).values
        )
        outcomes.append(
            explain_outcomes(
                agent, split.test, background, normalizer=split.normalizer,
                cfg=OutcomeAttributionConfig(
                    n_coalitions=160, n_episodes=300, seed=args.seed
                ),
            ).values
        )

    rows = [
        summarise("behaviour, pi(a|s)", behaviour),
        summarise("outcomes, episode return", outcomes),
    ]

    print(f"\n  {'target':<26}{'rank corr':>12}{'min':>8}"
          f"{'top-1 agree':>13}{'sign agree':>12}")
    for r in rows:
        print(f"  {r['target']:<26}{r['rank_corr_mean']:>+12.3f}"
              f"{r['rank_corr_min']:>+8.3f}{r['top1_agreement']:>13.0%}"
              f"{r['sign_agreement']:>12.0%}")

    print("\n  a rank correlation near 1.0 would mean explanations are a property")
    print("  of the task. near 0 would mean they are a property of the seed.")

    banner("3. IS OUTCOME-BASED ATTRIBUTION MORE STABLE THAN BEHAVIOUR-BASED?")
    b, o = rows[0]["rank_corr_mean"], rows[1]["rank_corr_mean"]
    print(f"  behaviour  {b:+.3f}")
    print(f"  outcomes   {o:+.3f}")
    if abs(o - b) < 0.05:
        verdict = "no meaningful difference"
    elif o > b:
        verdict = "outcomes MORE stable"
    else:
        verdict = "outcomes LESS stable"
    print(f"  -> {verdict}")

    # one bar per feature per seed would be unreadable; plot the spread instead
    plots.attribution_stability(
        list(FEATURE_NAMES),
        np.array(behaviour),
        np.array(outcomes),
        out / "attribution_stability.png",
        subtitle=f"{len(agents)} seeds, {n_indist}/{len(pairs)} pairs "
                 f"statistically indistinguishable in performance",
    )

    payload = {
        "n_seeds": len(agents),
        "seed_pnl": [float(p.mean()) for p in pnls],
        "performance_pvalues": [round(p, 4) for p in ps],
        "n_indistinguishable_pairs": n_indist,
        "n_pairs": len(pairs),
        "stability": rows,
        "verdict": verdict,
    }
    (out / "stability.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/stability.json")


if __name__ == "__main__":
    main()
