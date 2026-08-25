"""train ppo on the real kalshi corpus, across multiple seeds.

trains on the train split, selects on val, and does NOT touch test. test is
evaluated once, by scripts/evaluate.py, after everything is frozen.

each seed writes a checkpoint and a json log so that learning curves can be
regenerated without retraining. per the spec, results are reported as mean +/-
std across seeds rather than as a single run.

usage:
    python scripts/train_ppo.py --corpus data/corpus/corpus_candles_60s_spot.npz \\
        --seeds 5 --updates 150 --out runs/ppo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.costs import CostModel  # noqa: E402
from nano_rl.env.features import N_FEATURES  # noqa: E402


def train_one_seed(
    split,
    seed: int,
    updates: int,
    entropy_coef: float,
    episodes_per_batch: int,
    max_position: float,
    frictionless: bool,
    out_dir: Path,
) -> dict:
    """train one seed and return its val summary."""
    cost = CostModel(enabled=not frictionless)

    train_env = BinaryMarketEnv(
        split.train,
        cost_model=cost,
        normalizer=split.normalizer,
        max_position=max_position,
    )
    train_env.reset(seed=seed)

    cfg = PPOConfig(
        seed=seed, entropy_coef=entropy_coef, episodes_per_batch=episodes_per_batch
    )
    agent = PPOAgent(N_FEATURES, 3, cfg)

    tag = f"seed{seed}" + ("_nofric" if frictionless else "")
    print(f"\n--- {tag} ---", flush=True)
    t0 = time.time()
    log = agent.train(train_env, n_updates=updates, log_every=max(updates // 8, 1))

    # val is used for model selection and sanity, never for reporting
    val_env = BinaryMarketEnv(
        split.val,
        cost_model=cost,
        normalizer=split.normalizer,
        max_position=max_position,
        random_episode_order=False,
    )
    val = agent.evaluate(val_env, n_episodes=len(split.val))

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / f"{tag}.pt"))
    (out_dir / f"{tag}_log.json").write_text(json.dumps(log.as_dict()))

    summary = {
        "seed": seed,
        "frictionless": frictionless,
        "val_return_mean": float(val["returns"].mean()),
        "val_return_std": float(val["returns"].std()),
        "val_trades_mean": float(val["trades"].mean()),
        "val_fees_mean": float(val["fees"].mean()),
        "final_entropy": float(log.entropy[-1]),
        "final_explained_var": float(log.explained_var[-1]),
        "final_action_freq": list(log.action_freq[-1]),
        "train_seconds": round(time.time() - t0, 1),
    }
    print(
        f"  val: {summary['val_return_mean']:+.3f} +/- {summary['val_return_std']:.2f} "
        f"| trades {summary['val_trades_mean']:.2f} "
        f"| entropy {summary['final_entropy']:.3f} "
        f"| {summary['train_seconds']:.0f}s",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--entropy-coef", type=float, default=0.05)
    ap.add_argument("--episodes-per-batch", type=int, default=64)
    ap.add_argument("--max-position", type=float, default=100.0)
    ap.add_argument(
        "--frictionless",
        action="store_true",
        help="zero-cost ablation: separates 'cannot predict' from "
        "'predicts but cannot cover costs' (docs/MDP.md section 9.4)",
    )
    args = ap.parse_args()

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)

    print("=" * 72)
    print(f"corpus  : {args.corpus}")
    print(f"episodes: {len(batch)}  steps/ep: {batch.n_steps}  spot: {batch.has_spot}")
    print(split.summary())
    print(f"config  : {args.seeds} seeds, {args.updates} updates, "
          f"entropy {args.entropy_coef}, batch {args.episodes_per_batch}")
    print(f"costs   : {'DISABLED (ablation)' if args.frictionless else 'enabled'}")
    print("=" * 72)

    out_dir = Path(args.out)
    summaries = [
        train_one_seed(
            split,
            seed,
            args.updates,
            args.entropy_coef,
            args.episodes_per_batch,
            args.max_position,
            args.frictionless,
            out_dir,
        )
        for seed in range(args.seeds)
    ]

    rets = np.array([s["val_return_mean"] for s in summaries])
    trades = np.array([s["val_trades_mean"] for s in summaries])

    print("\n" + "=" * 72)
    print("across seeds (VAL split; test remains untouched)")
    print("=" * 72)
    print(f"  return : {rets.mean():+.3f} +/- {rets.std():.3f}")
    print(f"  trades : {trades.mean():.2f} +/- {trades.std():.2f}")
    print(f"  per-seed returns: {[round(float(r), 3) for r in rets]}")

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
