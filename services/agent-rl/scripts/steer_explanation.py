"""can an agent's explanation be changed while its performance is held fixed?

this is the safety-relevant question in the project. if an explanation can be
steered without cost, then it reflects a training choice rather than the task,
and an overseer inspecting attributions is inspecting something the developer
controls. the threat model is concrete: a developer who wanted their agent to
appear not to use a sensitive feature could arrange it, and standard attribution
inspection would not notice.

the method is an auxiliary penalty during ppo training on the divergence
between pi(.|s) and pi(.|s') where the target feature is resampled from the
batch marginal. that is the same interventional perturbation shapley
attribution measures, so the penalty attacks the attribution directly.

THE CONTROL IS THE POINT. the experiment is run on two corpora:

  real market        the target feature dominates the explanation but the
                     agent has no measurable edge. prediction: steering
                     succeeds, since nothing load-bearing is being removed.

  learnable synthetic  the target feature is the planted signal and is
                     genuinely necessary. prediction: steering FAILS, because
                     removing it costs return.

if both hold, the finding is sharp and falsifiable: **explanations are steerable
exactly when they are not tracking anything real.** that ties directly to the
null test, which says the real agent's explanation is not tracking anything.

if steering also succeeds on the synthetic corpus, the result is weaker and
different: it would mean the penalty is defeating the attribution method rather
than changing what the agent uses, and the report must say so.

usage:
    python scripts/steer_explanation.py --corpus data/corpus/corpus_candles_60s_spot.npz
"""

from __future__ import annotations

import argparse
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
from nano_rl.env.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    SIGNAL_OBS_IDX,
    feature_index,
    fit_normalizer,
)
from nano_rl.env.synthetic import make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.trajectory import explain_behaviour  # noqa: E402
from nano_rl.metrics import paired_bootstrap_p_value  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def global_attribution(
    agent: PPOAgent, env: BinaryMarketEnv, background: np.ndarray,
    n_states: int = 25, n_permutations: int = 50, seed: int = 0,
) -> np.ndarray:
    """mean absolute per-decision attribution over sampled states."""
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
    return acc / n_states


def run_condition(
    train_batch, eval_batch, norm, target: int, coef: float,
    updates: int, seeds: int, n_states: int, base_seed: int,
) -> dict:
    """train `seeds` agents at one penalty strength; measure both quantities."""
    returns, shares = [], []
    all_pnl = []

    for s in range(seeds):
        env = BinaryMarketEnv(train_batch, max_position=100.0, normalizer=norm)
        env.reset(seed=base_seed + s)
        cfg = PPOConfig(
            seed=base_seed + s,
            invariance_feature=target if coef > 0 else None,
            invariance_coef=coef,
        )
        agent = PPOAgent(N_FEATURES, 3, cfg)
        agent.train(env, n_updates=updates, verbose=False)

        ev = BinaryMarketEnv(
            eval_batch, max_position=100.0, normalizer=norm,
            random_episode_order=False,
        )
        res = agent.evaluate(ev, n_episodes=min(500, len(eval_batch)))
        returns.append(float(res["returns"].mean()))
        all_pnl.append(res["returns"])

        roll = VectorizedRollout(eval_batch, normalizer=norm, max_position=100.0)
        bg = build_background(roll, n_samples=192, seed=base_seed + s)
        att = global_attribution(
            agent, ev, bg, n_states=n_states, seed=base_seed + s
        )
        total = att.sum()
        shares.append(float(att[target] / total) if total > 0 else 0.0)

    return {
        "coef": coef,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "target_share_mean": float(np.mean(shares)),
        "target_share_std": float(np.std(shares)),
        "pnl": all_pnl,
    }


def run_corpus(
    name: str, train_batch, eval_batch, norm, target: int,
    coefs: list[float], updates: int, seeds: int, n_states: int,
    base_seed: int, prediction: str,
) -> list[dict]:
    banner(f"{name}   target feature: `{FEATURE_NAMES[target]}`")
    print(f"  prediction: {prediction}\n")
    print(f"  {'penalty':>9} {'return':>18} {'attribution share':>20} "
          f"{'p vs baseline':>14}")

    rows = []
    baseline_pnl = None
    for c in coefs:
        r = run_condition(
            train_batch, eval_batch, norm, target, c,
            updates, seeds, n_states, base_seed,
        )
        if baseline_pnl is None:
            baseline_pnl = r["pnl"]
            p = float("nan")
        else:
            # paired bootstrap on matched episodes, seed by seed, then pooled
            ps = [
                paired_bootstrap_p_value(a, b)
                for a, b in zip(r["pnl"], baseline_pnl)
            ]
            p = float(np.median(ps))
        r["p_vs_baseline"] = p
        r.pop("pnl")
        rows.append(r)

        print(
            f"  {c:>9.1f} {r['return_mean']:>+10.2f} +/-{r['return_std']:<5.2f} "
            f"{r['target_share_mean']:>14.1%} +/-{r['target_share_std']:<4.1%} "
            f"{p:>14.3f}",
            flush=True,
        )

    base = rows[0]
    best = min(rows[1:], key=lambda r: r["target_share_mean"]) if len(rows) > 1 else base
    drop = 1.0 - (best["target_share_mean"] / max(base["target_share_mean"], 1e-9))
    perf_ok = best["p_vs_baseline"] >= 0.05

    print(f"\n  attribution to `{FEATURE_NAMES[target]}` fell "
          f"{base['target_share_mean']:.1%} -> {best['target_share_mean']:.1%} "
          f"({drop:.0%} reduction)")
    print(f"  performance {'held' if perf_ok else 'CHANGED'} "
          f"(p = {best['p_vs_baseline']:.3f} vs baseline)")
    verdict = "STEERABLE" if (drop > 0.5 and perf_ok) else "NOT STEERABLE WITHOUT COST"
    print(f"  -> {verdict}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--coefs", type=float, nargs="+",
                    default=[0.0, 1.0, 5.0, 20.0])
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-states", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- the real market
    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    target_real = feature_index("time_to_expiry_frac")

    real_rows = run_corpus(
        "REAL MARKET (agent has no measurable edge)",
        split.train, split.test, split.normalizer, target_real,
        args.coefs, args.updates, args.seeds, args.n_states, args.seed,
        "steering should SUCCEED: nothing load-bearing is being removed",
    )

    # ---------------------------------------------------------- the control
    lb = make_learnable_corpus(n_episodes=1500, seed=args.seed)
    ln = fit_normalizer(lb)

    synth_rows = run_corpus(
        "LEARNABLE SYNTHETIC (the target feature IS the signal)",
        lb, lb, ln, SIGNAL_OBS_IDX,
        args.coefs, args.updates, args.seeds, args.n_states, args.seed,
        "steering should FAIL: removing the signal must cost return",
    )

    # -------------------------------------------------------------- verdict
    banner("WHAT THIS MEANS")

    def summarise(rows):
        base, best = rows[0], min(rows[1:], key=lambda r: r["target_share_mean"])
        drop = 1.0 - (best["target_share_mean"] / max(base["target_share_mean"], 1e-9))
        return drop, best["p_vs_baseline"], base["return_mean"], best["return_mean"]

    rd, rp, rb0, rb1 = summarise(real_rows)
    sd, sp, sb0, sb1 = summarise(synth_rows)

    print(f"  {'corpus':<26}{'attr drop':>11}{'return':>22}{'p':>8}")
    print(f"  {'real market':<26}{rd:>10.0%}  {rb0:>+8.2f} -> {rb1:>+8.2f}{rp:>8.3f}")
    print(f"  {'learnable synthetic':<26}{sd:>10.0%}  {sb0:>+8.2f} -> {sb1:>+8.2f}{sp:>8.3f}")

    steerable_real = rd > 0.5 and rp >= 0.05
    steerable_synth = sd > 0.5 and sp >= 0.05

    print()
    if steerable_real and not steerable_synth:
        print("  the explanation is steerable on the real market and NOT on the")
        print("  corpus where the feature genuinely matters. so explanations are")
        print("  steerable exactly where they are not tracking anything real,")
        print("  which is precisely what the null test says of the real agent.")
    elif steerable_real and steerable_synth:
        print("  steering succeeded on BOTH, including where the feature is")
        print("  genuinely load-bearing. that is a weaker and different result:")
        print("  the penalty is defeating the attribution method rather than")
        print("  changing what the agent relies on. reported as such.")
    elif not steerable_real:
        print("  steering did not succeed even on the real market. the")
        print("  explanation resists being changed at fixed performance, which")
        print("  is evidence it tracks something the penalty cannot remove.")

    plots.steering(
        {"real market": real_rows, "learnable synthetic": synth_rows},
        out / "explanation_steering.png",
        target_names={
            "real market": FEATURE_NAMES[target_real],
            "learnable synthetic": FEATURE_NAMES[SIGNAL_OBS_IDX],
        },
    )

    (out / "steering.json").write_text(json.dumps(
        {"real_market": real_rows, "learnable_synthetic": synth_rows,
         "steerable_real": bool(steerable_real),
         "steerable_synthetic": bool(steerable_synth)},
        indent=2,
    ))
    print(f"\nwrote {out}/steering.json")


if __name__ == "__main__":
    main()
