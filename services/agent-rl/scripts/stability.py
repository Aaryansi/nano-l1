"""Describe variation in feature attributions across independently trained seeds.

Report every seed and every pair. Paired UTC-day bootstrap comparisons describe
mean-return differences; non-rejection does not establish equal performance or
justify filtering the attribution comparisons. Rank, top-feature and signed
agreement are distinct diagnostics, conditional on the attribution procedure.
They do not establish that a ranking is a property of the task.
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
from nano_rl.explain.diagnostics import magnitude_rank_correlation  # noqa: E402
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
) -> tuple[np.ndarray, np.ndarray]:
    """mean absolute per-decision attribution over sampled states.

    a single-state attribution is dominated by where that state happens to sit
    relative to the background, so the global summary is what gets compared.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros(N_FEATURES)
    signed = np.zeros(N_FEATURES)

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
        signed += att.values

    return acc / n_states, signed / n_states


rank_correlation = magnitude_rank_correlation


def summarise(name: str, atts: list[np.ndarray], signed_atts=None) -> dict:
    """pairwise agreement statistics across seeds."""
    pairs = list(itertools.combinations(range(len(atts)), 2))
    rhos, top1, signs = [], [], []

    for i, j in pairs:
        a, b = atts[i], atts[j]
        rho = rank_correlation(a, b)
        if np.isfinite(rho):
            rhos.append(rho)
        if (np.abs(a).max() > 1e-12 and np.abs(b).max() > 1e-12
                and np.count_nonzero(np.abs(a) == np.abs(a).max()) == 1
                and np.count_nonzero(np.abs(b) == np.abs(b).max()) == 1):
            top1.append(int(np.argmax(np.abs(a))) == int(np.argmax(np.abs(b))))

        sa, sb = (a, b) if signed_atts is None else (signed_atts[i], signed_atts[j])
        # sign agreement only where BOTH assign meaningful mass, since the sign
        # of a negligible attribution is noise
        thresh_a = 0.1 * np.abs(sa).max()
        thresh_b = 0.1 * np.abs(sb).max()
        both = (np.abs(sa) > max(thresh_a, 1e-12)) & (np.abs(sb) > max(thresh_b, 1e-12))
        if both.sum() > 0:
            signs.append(float((np.sign(sa[both]) == np.sign(sb[both])).mean()))

    return {
        "target": name,
        "n_pairs": len(pairs),
        "n_defined_rank_pairs": len(rhos),
        "n_unique_top_pairs": len(top1),
        "rank_corr_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "rank_corr_min": float(np.min(rhos)) if rhos else float("nan"),
        "rank_corr_std": float(np.std(rhos)) if rhos else float("nan"),
        "top1_agreement": float(np.mean(top1)) if top1 else float("nan"),
        "sign_agreement": float(np.mean(signs)) if signs else float("nan"),
        "per_pair_rho": [round(r, 3) for r in rhos],
    }


def compare_stability(behaviour, outcomes):
    if not np.isfinite(behaviour) or not np.isfinite(outcomes):
        return "undefined rank comparison"
    if abs(outcomes - behaviour) < .05:
        return "mean rank correlations differ by less than 0.05 (descriptive threshold)"
    return "outcomes have higher mean rank correlation" if outcomes > behaviour else "outcomes have lower mean rank correlation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-states", type=int, default=30)
    ap.add_argument("--outcome-coalitions", type=int, default=160)
    ap.add_argument("--outcome-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if min(args.n_states, args.outcome_coalitions, args.outcome_episodes) < 1:
        raise SystemExit("attribution budgets must be positive")

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

    banner(f"1. PERFORMANCE DIFFERENCES ACROSS {len(agents)} SEEDS")
    print("  approximate paired UTC-day bootstrap; no equivalence claim or pair filtering")

    env = BinaryMarketEnv(
        split.test, normalizer=split.normalizer, max_position=100.0,
        random_episode_order=False,
    )
    pnls = [a.evaluate(env, n_episodes=len(split.test))["returns"] for a in agents]

    print(f"  {'seed':>5} {'mean pnl':>10}")
    for i, p in enumerate(pnls):
        print(f"  {i:>5} {p.mean():>+10.4f}")

    pairs = list(itertools.combinations(range(len(agents)), 2))
    blocks = np.floor(split.test.open_epoch / 86400).astype(int)
    ps = [paired_bootstrap_p_value(pnls[i], pnls[j], seed=args.seed, block_ids=blocks) for i, j in pairs]
    n_not_detected = sum(1 for p in ps if p >= 0.05)
    print(f"\n  pairwise paired UTC-day bootstrap p-values: "
          f"min {min(ps):.3f}, median {float(np.median(ps)):.3f}")
    print(f"  {n_not_detected} of {len(pairs)} pairs have no detected mean-return difference")
    print(f"  {len(np.unique(blocks))} UTC-day blocks; small block/seed counts limit inference")

    banner("2. DO THEY EXPLAIN THEMSELVES THE SAME WAY?")

    roll = VectorizedRollout(
        split.test, normalizer=split.normalizer, max_position=100.0
    )
    background = build_background(roll, n_samples=256, seed=args.seed)

    behaviour, outcomes, signed_behaviour = [], [], []
    for i, agent in enumerate(agents):
        print(f"  seed {i}: attributing...", flush=True)
        magnitude, signed = global_behaviour_attribution(
            agent, env, background, n_states=args.n_states, seed=args.seed
        )
        behaviour.append(magnitude)
        signed_behaviour.append(signed)
        outcomes.append(
            explain_outcomes(
                agent, split.test, background, normalizer=split.normalizer,
                cfg=OutcomeAttributionConfig(
                    n_coalitions=args.outcome_coalitions, n_episodes=args.outcome_episodes, seed=args.seed
                ),
            ).values
        )

    rows = [
        summarise("behaviour, pi(a|s)", behaviour, signed_behaviour),
        summarise("outcomes, episode return", outcomes),
    ]

    print(f"\n  {'target':<26}{'rank corr':>12}{'min':>8}"
          f"{'top-1 agree':>13}{'sign agree':>12}")
    for r in rows:
        print(f"  {r['target']:<26}{r['rank_corr_mean']:>+12.3f}"
              f"{r['rank_corr_min']:>+8.3f}{r['top1_agreement']:>13.0%}"
              f"{r['sign_agreement']:>12.0%}")

    print("\n  rank agreement describes these agents and this attribution procedure.")

    banner("3. IS OUTCOME-BASED ATTRIBUTION MORE STABLE THAN BEHAVIOUR-BASED?")
    b, o = rows[0]["rank_corr_mean"], rows[1]["rank_corr_mean"]
    print(f"  behaviour  {b:+.3f}")
    print(f"  outcomes   {o:+.3f}")
    verdict = compare_stability(b, o)
    print(f"  -> {verdict}")

    # one bar per feature per seed would be unreadable; plot the spread instead
    plots.attribution_stability(
        list(FEATURE_NAMES),
        np.array(behaviour),
        np.array(outcomes),
        out / "attribution_stability.png",
        subtitle=f"{len(agents)} seeds, {n_not_detected}/{len(pairs)} pairs "
                 f"without a detected return difference; no equivalence claim",
    )

    payload = {
        "n_seeds": len(agents),
        "seed_pnl": [float(p.mean()) for p in pnls],
        "performance_pvalues": ps,
        "performance_pairs": pairs,
        "performance_returns_file": "stability_returns.npz",
        "performance_bootstrap_seed": args.seed,
        "performance_bootstrap_resamples": 10000,
        "performance_utc_day_blocks": len(np.unique(blocks)),
        "performance_scope": "paired UTC-day bootstrap conditional on fitted agents; no multiplicity correction or equivalence claim",
        "n_pairs_without_detected_difference": n_not_detected,
        "n_pairs": len(pairs),
        "stability": [{key: (value if not isinstance(value, float) or np.isfinite(value) else None)
                       for key, value in row.items()} for row in rows],
        "outcome_attribution_episodes": min(args.outcome_episodes, len(split.test)),
        "outcome_coalitions": args.outcome_coalitions,
        "behaviour_states": args.n_states,
        "behaviour_magnitudes": [x.tolist() for x in behaviour],
        "behaviour_signed_means": [x.tolist() for x in signed_behaviour],
        "outcome_values": [x.tolist() for x in outcomes],
        "sign_agreement_estimand": "sign of mean signed attribution on features above 10% of each run maximum",
        "verdict": verdict,
    }
    np.savez_compressed(out / "stability_returns.npz", returns=np.stack(pnls), open_epoch=split.test.open_epoch)
    (out / "stability.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"\nwrote {out}/stability.json")


if __name__ == "__main__":
    main()
