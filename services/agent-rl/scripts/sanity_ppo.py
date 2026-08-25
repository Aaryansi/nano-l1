"""sanity check: is the ppo implementation correct?

same two synthetic corpora as scripts/sanity_tabular.py, so ppo is held to the
benchmark tabular q already cleared. tabular q reached 98.2% of optimum on the
learnable corpus and exactly 0.00 on the null corpus, which means the
environment is sound and any ppo failure here is ppo's.

both must hold:
    learnable  approach signal_policy_return
    null       abstain, ~0 return and few trades

usage:
    python scripts/sanity_ppo.py --updates 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import (  # noqa: E402
    flat_policy_return,
    make_learnable_corpus,
    make_null_corpus,
    signal_policy_return,
)


def run_case(name: str, batch, benchmark: float, updates: int, seed: int, expect: str) -> dict:
    print(f"\n{'=' * 74}")
    print(f"{name}   (expect: {expect})")
    print("=" * 74)

    # normalising is not optional for a neural agent here: raw observations
    # reach magnitude 9.2 on volume_rate, which saturated 31% of the first
    # tanh layer and pinned ppo at 0.00 return. see scripts/diagnose_ppo.py.
    norm = fit_normalizer(batch)

    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    log = agent.train(env, n_updates=updates, log_every=max(updates // 6, 1))

    eval_env = BinaryMarketEnv(
        batch, max_position=100.0, normalizer=norm, random_episode_order=False
    )
    res = agent.evaluate(eval_env, n_episodes=500)

    print(f"\n  benchmark          : {benchmark:>8.2f} / episode")
    print(f"  greedy policy      : {res['returns'].mean():>8.2f} "
          f"+/- {res['returns'].std():.2f}")
    print(f"  trades per episode : {res['trades'].mean():>8.2f}  (of 14 steps)")
    print(f"  mean fees          : {res['fees'].mean():>8.2f}")
    print(f"  explained variance : {log.explained_var[-1]:>8.3f}  (critic quality)")
    print(f"  final entropy      : {log.entropy[-1]:>8.3f}  (ln 3 = 1.099 is uniform)")
    s, f, l = log.action_freq[-1]
    print(f"  action mix S/F/L   : {s:.2f} / {f:.2f} / {l:.2f}")

    return {
        "benchmark": benchmark,
        "greedy": float(res["returns"].mean()),
        "trades": float(res["trades"].mean()),
        "explained_var": float(log.explained_var[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lb = make_learnable_corpus(n_episodes=2000, seed=args.seed)
    learn = run_case(
        "LEARNABLE corpus",
        lb,
        signal_policy_return(lb, max_position=100.0),
        args.updates,
        args.seed,
        "approach the benchmark",
    )

    nb = make_null_corpus(n_episodes=2000, seed=args.seed + 1)
    null = run_case(
        "NULL corpus",
        nb,
        flat_policy_return(),
        args.updates,
        args.seed,
        "abstain, ~0, few trades",
    )

    print(f"\n{'=' * 74}")
    print("verdict")
    print("=" * 74)

    ok_learn = learn["greedy"] > 0.5 * learn["benchmark"]
    ok_null = null["greedy"] > -3.0 and null["trades"] < 2.0

    print(f"  learnable: {learn['greedy']:.2f} vs {learn['benchmark']:.2f} "
          f"({learn['greedy']/learn['benchmark']:.1%})  -> {'PASS' if ok_learn else 'FAIL'}")
    print(f"  null     : {null['greedy']:.2f} vs 0.00, {null['trades']:.2f} trades/ep "
          f"-> {'PASS' if ok_null else 'FAIL'}")

    if not (ok_learn and ok_null):
        print("\nPPO SANITY FAILED: the env is known good, so this is the agent")
        raise SystemExit(1)
    print("\nppo implementation is sound on both known-answer corpora")


if __name__ == "__main__":
    main()
