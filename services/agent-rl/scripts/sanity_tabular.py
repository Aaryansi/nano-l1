"""sanity check: can a tabular learner solve a corpus with known signal?

this must pass before any ppo result is believable. it validates the
environment, the reward accounting, and the feature plumbing end to end using
the simplest possible learner, with no function approximation, replay, or
target network to mask an environment bug.

two cases, each with its OWN benchmark:

    learnable corpus  benchmark is signal_policy_return, which is genuinely
                      optimal there. the agent must approach it. failure means
                      the signal cannot reach the agent.

    null corpus       benchmark is flat_policy_return, i.e. exactly zero.
                      signal_policy_return is the WRONG target here, since
                      following noise just pays frictions. the agent must
                      learn to abstain.

on the training budget: the null case needs materially more episodes than the
learnable one, and the reason is measured in scripts/diagnose_null.py. true
action values on the null corpus differ by 2.75 while single-episode returns
have a standard deviation near 50, a signal-to-noise ratio of about 1:18. until
the frequently-visited bins agree on FLAT, the binned signal re-randomises each
step and the position churns. 3000 episodes is not reliably enough; 20000 is.

usage:
    python scripts/sanity_tabular.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.tabular_q import TabularQAgent, TabularQConfig  # noqa: E402
from nano_rl.env.binary_market import ACTION_NAMES, BinaryMarketEnv  # noqa: E402
from nano_rl.env.features import KALSHI_FEATURES, SIGNAL_OBS_IDX  # noqa: E402
from nano_rl.env.synthetic import (  # noqa: E402
    flat_policy_return,
    make_learnable_corpus,
    make_null_corpus,
    signal_policy_return,
)


def run_case(name: str, batch, benchmark: float, n_train: int, seed: int, expect: str) -> dict:
    env = BinaryMarketEnv(batch, max_position=100.0)
    agent = TabularQAgent(
        TabularQConfig(seed=seed, eps_decay_episodes=max(n_train // 4, 1))
    )
    log = agent.train(env, n_episodes=n_train, signal_idx=SIGNAL_OBS_IDX)

    eval_env = BinaryMarketEnv(batch, max_position=100.0, random_episode_order=False)
    greedy = agent.evaluate(eval_env, n_episodes=500, signal_idx=SIGNAL_OBS_IDX)

    # how often is a position actually taken? churn is the failure mode on the
    # null corpus, so it is reported rather than inferred from the return.
    trades = []
    for ep in range(200):
        obs, _ = eval_env.reset(options={"episode": ep})
        info = {}
        while True:
            from nano_rl.env.synthetic import discretize

            s = discretize(obs, SIGNAL_OBS_IDX, agent.cfg.n_bins)
            obs, _, done, _, info = eval_env.step(agent.act(s, greedy=True))
            if done:
                break
        trades.append(info["trades"])

    print(f"\n{'=' * 68}")
    print(f"{name}   (expect: {expect})")
    print("=" * 68)
    print(f"  benchmark             : {benchmark:>8.2f} / episode")
    print(f"  greedy policy         : {greedy.mean():>8.2f} +/- {greedy.std():.2f}")
    print(f"  trades per episode    : {np.mean(trades):>8.2f}  (of 14 steps)")
    print(f"  train return first/last 50: {np.mean(log.returns[:50]):>7.2f} "
          f"/ {np.mean(log.returns[-50:]):.2f}")
    print(f"  mean |td| first/last 50   : {np.mean(log.td_errors[:50]):>7.3f} "
          f"/ {np.mean(log.td_errors[-50:]):.3f}")
    print(f"  final epsilon         : {log.epsilons[-1]:>8.3f}")
    greedy_actions = np.argmax(agent.q, axis=1)
    print(f"  greedy action by bin  : {''.join(ACTION_NAMES[a][0] for a in greedy_actions)}")

    return {
        "benchmark": benchmark,
        "greedy": float(greedy.mean()),
        "trades": float(np.mean(trades)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn-episodes", type=int, default=3000)
    ap.add_argument("--null-episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"observation layout: signal at index {SIGNAL_OBS_IDX} "
          f"(after {len(KALSHI_FEATURES)} kalshi features)")

    lb = make_learnable_corpus(n_episodes=2000, seed=args.seed)
    learn = run_case(
        "LEARNABLE corpus (signal present)",
        lb,
        signal_policy_return(lb, max_position=100.0),
        args.learn_episodes,
        args.seed,
        "approach the benchmark",
    )

    nb = make_null_corpus(n_episodes=2000, seed=args.seed + 1)
    null = run_case(
        "NULL corpus (no signal)",
        nb,
        flat_policy_return(),
        args.null_episodes,
        args.seed,
        "abstain, ~0, few trades",
    )

    print(f"\n{'=' * 68}")
    print("verdict")
    print("=" * 68)

    ok_learn = learn["greedy"] > 0.5 * learn["benchmark"]
    # tolerance covers residual churn from rarely-visited tail bins
    ok_null = null["greedy"] > -2.0 and null["trades"] < 1.0

    print(f"  learnable: {learn['greedy']:.2f} vs benchmark {learn['benchmark']:.2f} "
          f"({learn['greedy']/learn['benchmark']:.1%})  -> {'PASS' if ok_learn else 'FAIL'}")
    print(f"  null     : {null['greedy']:.2f} vs benchmark 0.00, "
          f"{null['trades']:.2f} trades/ep  -> {'PASS' if ok_null else 'FAIL'}")

    if not (ok_learn and ok_null):
        print("\nSANITY CHECK FAILED: do not trust any ppo result until this passes")
        raise SystemExit(1)
    print("\nboth pass: env, reward accounting, and feature plumbing are sound")


if __name__ == "__main__":
    main()
