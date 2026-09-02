"""are the steered and unsteered agents the same policy, or only equally bad?

scripts/steer_explanation.py shows that an auxiliary penalty drives a feature's
attribution share from 39.9% to 3.2% at no measurable cost in return. the paper
drew from that the conclusion "attribution is not identified by behaviour".

return is not behaviour. two policies can earn the same and act differently,
and on a market where nothing beats abstention they can both earn nothing while
disagreeing everywhere. so the claim as stated needs the stronger measurement,
which is this script: on identical held-out states, how often do the two agents
choose the same action, and how far apart are their action distributions?

if they agree, "not identified by behaviour" holds as written. if they do not,
the honest claim is the weaker "not identified by task performance", and the
paper should say that instead. the point of running this is that we do not know
which until we measure.

reported per seed pair, on the held-out test split only:
    greedy action agreement          fraction of states with the same argmax
    mean KL(p_base || p_steered)     nats, over the categorical action dist
    jensen-shannon divergence        symmetric, bounded, easier to read
    action marginals                 whether the mix of actions shifts
    mean |position| and trade count  whether the induced trajectories differ

usage:
    python scripts/behavioural_equivalence.py --corpus <path> --out reports
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
from nano_rl.env.features import N_FEATURES, feature_index  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train_one(batch, norm, target: int, coef: float, updates: int,
              seed: int) -> PPOAgent:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    cfg = PPOConfig(
        seed=seed,
        invariance_feature=target if coef > 0 else None,
        invariance_coef=coef,
    )
    agent = PPOAgent(N_FEATURES, 3, cfg)
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def action_distributions(agent: PPOAgent, obs: np.ndarray) -> np.ndarray:
    """softmax action probabilities for a batch of observations."""
    with torch.no_grad():
        logits, _ = agent.net(torch.as_tensor(obs, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).numpy()


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * np.log(p / q)).sum(axis=-1)


def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def rollout_stats(agent: PPOAgent, batch, norm, n_episodes: int) -> dict:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm,
                          random_episode_order=False)
    res = agent.evaluate(env, n_episodes=n_episodes)
    return {
        "return_mean": float(res["returns"].mean()),
        "trades_per_episode": float(np.mean(res["n_trades"]))
        if "n_trades" in res else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--coef", type=float, default=20.0)
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-states", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    target = feature_index("time_to_expiry_frac")

    # held-out states only. the background sampler walks the test rollout and
    # returns real observations, which is exactly the state distribution the
    # comparison should be made on.
    roll = VectorizedRollout(split.test, normalizer=split.normalizer,
                             max_position=100.0)
    states = build_background(roll, n_samples=args.n_states, seed=args.seed)

    banner(f"behavioural equivalence, coef 0 vs {args.coef:g}, "
           f"{len(states)} held-out states")

    rows = []
    for s in range(args.seeds):
        base = train_one(split.train, split.normalizer, target, 0.0,
                         args.updates, args.seed + s)
        steer = train_one(split.train, split.normalizer, target, args.coef,
                          args.updates, args.seed + s)

        pb = action_distributions(base, states)
        ps = action_distributions(steer, states)
        agree = float((pb.argmax(1) == ps.argmax(1)).mean())
        row = {
            "seed": args.seed + s,
            "action_agreement": agree,
            "mean_kl": float(kl(pb, ps).mean()),
            "mean_js": float(js(pb, ps).mean()),
            "marginal_base": pb.mean(0).tolist(),
            "marginal_steered": ps.mean(0).tolist(),
            "base": rollout_stats(base, split.test, split.normalizer, 400),
            "steered": rollout_stats(steer, split.test, split.normalizer, 400),
        }
        rows.append(row)
        print(f"  seed {row['seed']}  agreement {agree:6.1%}   "
              f"KL {row['mean_kl']:.4f}   JS {row['mean_js']:.4f}")

    agreement = float(np.mean([r["action_agreement"] for r in rows]))
    mean_kl = float(np.mean([r["mean_kl"] for r in rows]))
    mean_js = float(np.mean([r["mean_js"] for r in rows]))

    # the threshold is a reporting convention, not a test. we set it before
    # looking, at the level a reader would accept as "the same policy".
    equivalent = bool(agreement >= 0.90 and mean_js <= 0.05)

    banner("verdict")
    print(f"  mean greedy action agreement : {agreement:.1%}")
    print(f"  mean KL(base || steered)     : {mean_kl:.4f} nats")
    print(f"  mean Jensen-Shannon          : {mean_js:.4f}")
    print(f"  behaviourally equivalent     : {equivalent}")
    if equivalent:
        print("\n  the steered agent acts like the unsteered one and explains")
        print("  differently, so 'not identified by behaviour' holds.")
    else:
        print("\n  the two agents act differently. the defensible claim is that")
        print("  attribution is not identified by task PERFORMANCE, and the")
        print("  paper should say that rather than 'behaviour'.")

    result = {
        "coef": args.coef,
        "n_states": len(states),
        "per_seed": rows,
        "action_agreement": agreement,
        "mean_kl": mean_kl,
        "mean_js": mean_js,
        "behaviourally_equivalent": equivalent,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "behavioural_equivalence.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out / 'behavioural_equivalence.json'}")


if __name__ == "__main__":
    main()
