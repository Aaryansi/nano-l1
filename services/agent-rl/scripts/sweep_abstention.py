"""can ppo learn to abstain on a no-signal corpus, and under what settings?

this is not a tuning exercise, it is a validity check on the whole project.
the real corpus has essentially no exploitable signal, so phase 4 will report
whatever the agent does there. if ppo is structurally incapable of learning
"do not trade", then a phase-4 result of "the agent traded and lost money"
would be an artefact of the agent rather than a fact about the market, and the
report would be wrong.

the failure being investigated: with 250 updates on a pure-noise corpus, ppo
converged to 96% LONG and -0.95 per episode, when FLAT earns exactly 0.00.
entropy collapsed to 0.177 well before the advantage estimates could average
out the +/-50 settlement noise, so the policy committed to the least-bad
trading action instead of to no action.

candidate remedies, each with a mechanism rather than a hope:
  entropy_coef      keeps the policy from committing early
  episodes_per_batch  reduces advantage noise per update
  anneal_lr off     avoids freezing whatever it happened to believe at the end

usage:
    python scripts/sweep_abstention.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import (  # noqa: E402
    make_learnable_corpus,
    make_null_corpus,
    signal_policy_return,
)


def evaluate_config(batch, norm, cfg: PPOConfig, updates: int) -> dict:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    agent = PPOAgent(N_FEATURES, 3, cfg)
    log = agent.train(env, n_updates=updates, verbose=False)

    eval_env = BinaryMarketEnv(
        batch, max_position=100.0, normalizer=norm, random_episode_order=False
    )
    res = agent.evaluate(eval_env, n_episodes=500)
    s, f, l = log.action_freq[-1]
    return {
        "ret": float(res["returns"].mean()),
        "trades": float(res["trades"].mean()),
        "entropy": float(log.entropy[-1]),
        "flat_frac": float(f),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    null = make_null_corpus(n_episodes=2000, seed=1)
    learn = make_learnable_corpus(n_episodes=2000, seed=0)
    null_norm = fit_normalizer(null)
    learn_norm = fit_normalizer(learn)
    bench = signal_policy_return(learn, max_position=100.0)

    ent_coefs = [0.01, 0.05, 0.2]
    batch_sizes = [64, 256]

    print("=" * 88)
    print(f"NULL corpus: can it learn to abstain?  (target: ret 0.00, trades 0.00)")
    print(f"{args.seeds} seeds each, {args.updates} updates")
    print("=" * 88)
    print(f"  {'ent':>5} {'eps/batch':>10} | {'return':>16} {'trades':>14} "
          f"{'entropy':>8} {'flat%':>7}")

    results = {}
    for ent, bs in itertools.product(ent_coefs, batch_sizes):
        rets, trades, ents, flats = [], [], [], []
        for seed in range(args.seeds):
            r = evaluate_config(
                null,
                null_norm,
                PPOConfig(seed=seed, entropy_coef=ent, episodes_per_batch=bs),
                args.updates,
            )
            rets.append(r["ret"])
            trades.append(r["trades"])
            ents.append(r["entropy"])
            flats.append(r["flat_frac"])
        results[(ent, bs)] = (np.mean(rets), np.mean(trades))
        print(f"  {ent:>5} {bs:>10} | {np.mean(rets):>8.2f} +/- {np.std(rets):<5.2f} "
              f"{np.mean(trades):>7.2f} +/-{np.std(trades):<5.2f} "
              f"{np.mean(ents):>8.3f} {np.mean(flats):>7.2f}")

    # the winner must ALSO still solve the learnable corpus, or we have simply
    # traded one failure for another.
    best = min(results, key=lambda k: abs(results[k][0]) + results[k][1])
    print(f"\n  best on null: entropy_coef={best[0]}, episodes_per_batch={best[1]}")

    print("\n" + "=" * 88)
    print(f"does that setting still solve the LEARNABLE corpus? (benchmark {bench:.2f})")
    print("=" * 88)
    for seed in range(args.seeds):
        r = evaluate_config(
            learn,
            learn_norm,
            PPOConfig(seed=seed, entropy_coef=best[0], episodes_per_batch=best[1]),
            args.updates,
        )
        print(f"  seed {seed}: return {r['ret']:>8.2f} ({r['ret']/bench:>6.1%}), "
              f"trades {r['trades']:.2f}, entropy {r['entropy']:.3f}")


if __name__ == "__main__":
    main()
