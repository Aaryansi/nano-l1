"""why does tabular q churn on a corpus with no signal?

the null corpus has a trivially optimal policy: stay flat, earn exactly zero.
tabular q instead earned -18.29. this script separates the candidate causes so
the writeup can state the real one rather than a plausible one.

hypotheses:
  H1  sample efficiency. true action values differ by ~2.75 while single-episode
      returns have a std near 50, so the signal-to-noise per sample is ~1:18 and
      3000 episodes is simply not enough.
  H2  argmax instability. with all action values nearly equal, tiny estimation
      noise flips the greedy action between bins. since the binned signal
      re-randomises every step, the position then churns within an episode.
  H3  maximization bias. the max operator in the q backup biases values upward,
      so trading looks better than flat.

usage:
    python scripts/diagnose_null.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.tabular_q import TabularQAgent, TabularQConfig  # noqa: E402
from nano_rl.env.binary_market import ACTION_NAMES, BinaryMarketEnv  # noqa: E402
from nano_rl.env.costs import fee_dollars  # noqa: E402
from nano_rl.env.features import SIGNAL_OBS_IDX  # noqa: E402
from nano_rl.env.synthetic import make_null_corpus  # noqa: E402


def true_action_values(price: float = 0.5, spread: float = 0.02, q: float = 100.0):
    """analytic action values on the null corpus, where E[settlement] = 0.5."""
    bid, ask = price - spread / 2, price + spread / 2
    long_v = q * (0.5 - ask) - fee_dollars(q, ask)
    short_v = q * (bid - 0.5) - fee_dollars(q, bid)
    return {"SHORT": short_v, "FLAT": 0.0, "LONG": long_v}


def count_trades(agent, env, n=300) -> tuple[float, float]:
    """mean trades per episode and mean return under the greedy policy."""
    from nano_rl.env.synthetic import discretize

    trades, rets = [], []
    for ep in range(min(n, len(env.batch))):
        obs, _ = env.reset(options={"episode": ep})
        total, info = 0.0, {}
        while True:
            s = discretize(obs, SIGNAL_OBS_IDX, agent.cfg.n_bins)
            obs, r, done, _, info = env.step(agent.act(s, greedy=True))
            total += r
            if done:
                break
        trades.append(info["trades"])
        rets.append(total)
    return float(np.mean(trades)), float(np.mean(rets))


def main() -> None:
    print("=" * 70)
    print("analytic action values on the null corpus")
    print("=" * 70)
    tv = true_action_values()
    for k, v in tv.items():
        print(f"  {k:>6}: {v:+.3f}")
    spread_v = max(tv.values()) - min(tv.values())
    print(f"\n  best action is FLAT, by a margin of {spread_v:.3f}")
    print("  single-episode return std is ~50, so signal-to-noise per sample")
    print(f"  is roughly 1:{50 / spread_v:.0f}")

    batch = make_null_corpus(n_episodes=2000, seed=1)

    print("\n" + "=" * 70)
    print("H1: does more training fix it?")
    print("=" * 70)
    print(f"  {'episodes':>9} {'lr':>6} {'return':>9} {'trades/ep':>10}  greedy policy")
    for n_ep, lr in ((3_000, 0.1), (20_000, 0.1), (20_000, 0.01), (60_000, 0.01)):
        env = BinaryMarketEnv(batch, max_position=100.0)
        agent = TabularQAgent(TabularQConfig(seed=0, lr=lr, eps_decay_episodes=n_ep // 4))
        agent.train(env, n_episodes=n_ep, signal_idx=SIGNAL_OBS_IDX)

        eval_env = BinaryMarketEnv(batch, max_position=100.0, random_episode_order=False)
        t, r = count_trades(agent, eval_env)
        acts = [ACTION_NAMES[a][0] for a in np.argmax(agent.q, axis=1)]
        print(f"  {n_ep:>9} {lr:>6} {r:>9.2f} {t:>10.2f}  {''.join(acts)}")

    print("\n" + "=" * 70)
    print("H2: is the churn caused by the greedy action differing across bins?")
    print("=" * 70)
    env = BinaryMarketEnv(batch, max_position=100.0)
    agent = TabularQAgent(TabularQConfig(seed=0))
    agent.train(env, n_episodes=3000, signal_idx=SIGNAL_OBS_IDX)
    eval_env = BinaryMarketEnv(batch, max_position=100.0, random_episode_order=False)

    greedy = np.argmax(agent.q, axis=1)
    n_distinct = len(set(greedy.tolist()))
    t, r = count_trades(agent, eval_env)
    print(f"  distinct greedy actions across bins : {n_distinct}")
    print(f"  trades per episode                  : {t:.2f}  (14 steps per episode)")
    print(f"  return                              : {r:.2f}")
    print(f"  q-value spread within a typical bin : "
          f"{float(np.mean(agent.q.max(axis=1) - agent.q.min(axis=1))):.3f}")
    print(f"  (true spread between best and worst : {spread_v:.3f})")

    print("\n  forcing a CONSTANT action for comparison:")
    for a in range(3):
        const = TabularQAgent(TabularQConfig(seed=0))
        const.q[:] = 0.0
        const.q[:, a] = 1.0  # make action `a` greedy everywhere
        t2, r2 = count_trades(const, eval_env)
        print(f"    always {ACTION_NAMES[a]:<5}: return {r2:>8.2f}, trades {t2:.2f}")

    print("\n" + "=" * 70)
    print("H3: is the learned value of trading biased above its true value?")
    print("=" * 70)
    for a, name in enumerate(ACTION_NAMES):
        learned = float(agent.q[:, a].mean())
        print(f"  {name:>6}: learned {learned:+8.3f}   true {tv[name]:+8.3f}   "
              f"bias {learned - tv[name]:+8.3f}")


if __name__ == "__main__":
    main()
