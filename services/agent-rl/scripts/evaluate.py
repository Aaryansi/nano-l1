"""evaluate ppo and all baselines on the held-out test split.

Evaluate frozen policies. Test results are descriptive for the chosen training
configuration. Repeated analysis of this historical test split does not create
a new prospective holdout. Optional comparison checkpoints and serialized
baselines must be supplied explicitly; no local legacy files are discovered.

outputs to reports/: equity curves, policy comparison, learning curves,
critic calibration, cost ablation, and results.json.

usage:
    python scripts/evaluate.py \\
        --corpus data/corpus/corpus_candles_60s_spot.npz \\
        --runs runs/ppo --runs-nofric runs/ppo_nofric --out reports
"""

from __future__ import annotations

import argparse
import json
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl import plots  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.baselines import (  # noqa: E402
    Policy,
    default_baselines,
    fit_logistic_on_corpus,
)
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.costs import CostModel  # noqa: E402
from nano_rl.env.features import N_FEATURES  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout
from nano_rl.metrics import (  # noqa: E402
    Metrics,
    compute_metrics,
    paired_bootstrap_p_value,
    cluster_bootstrap_p_value,
    cluster_bootstrap_ci,
)


def run_policy(env: BinaryMarketEnv, policy: Policy, seed: int = 0) -> dict:
    """roll a baseline policy over every episode in the env's batch."""
    rng = np.random.default_rng(seed)
    pnl, trades, fees = [], [], []

    for ep in range(len(env.batch)):
        obs, _ = env.reset(options={"episode": ep})
        policy.reset()
        total, info = 0.0, {}
        while True:
            obs, r, done, _, info = env.step(policy.act(obs, rng))
            total += r
            if done:
                break
        pnl.append(total)
        trades.append(info["trades"])
        fees.append(info["fees"])

    return {
        "pnl": np.array(pnl),
        "trades": np.array(trades),
        "fees": np.array(fees),
    }


def run_agent(env: BinaryMarketEnv, agent: PPOAgent) -> dict:
    res = agent.evaluate(env, n_episodes=len(env.batch), deterministic=True)
    return {"pnl": res["returns"], "trades": res["trades"], "fees": res["fees"]}


def critic_calibration(agent, env, n_bins=10, repetitions=8, seed=0):
    """Compare V(s0) in dollars with stochastic on-policy episode returns.

    The PPO critic estimates its stochastic policy, not a forced long position
    and not the greedy evaluation policy. No probability transform is applied.
    """
    if repetitions < 1 or n_bins < 1:
        raise ValueError("repetitions and bins must be positive")
    roll = VectorizedRollout(env.batch, normalizer=env.normalizer,
                             max_position=env.max_position,
                             costs_enabled=env.cost_model.enabled)
    zeros = np.zeros(len(env.batch))
    initial = roll.observations(0, zeros, zeros, zeros)
    with torch.no_grad():
        preds = agent.net.value(torch.as_tensor(initial, dtype=torch.float32)).numpy()
    rng = np.random.default_rng(seed)
    def policy(obs):
        probs = agent.net.action_probs(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        return np.minimum((rng.random(len(obs))[:, None] > probs.cumsum(axis=1)).sum(axis=1), 2)
    draws = np.stack([roll.run(policy)["returns"] for _ in range(repetitions)])
    targets = draws.mean(axis=0)
    order = np.argsort(preds, kind="stable")
    bins = [(float(preds[idx].mean()), float(targets[idx].mean()), len(idx))
            for idx in np.array_split(order, min(n_bins, len(order))) if len(idx)]
    return {"bins": bins, "predictions": preds, "realized_returns": targets,
            "stochastic_return_draws": draws, "repetitions": repetitions}


def load_agents(run_dir: Path, tag: str = "") -> list[PPOAgent]:
    agents = []
    for p in sorted(run_dir.glob(f"seed*{tag}.pt")):
        a = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0))
        a.load(str(p))
        agents.append(a)
    return agents


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--runs-nofric", help="explicit zero-cost checkpoint directory")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--logistic-model", help="explicit optional serialized baseline")
    ap.add_argument("--max-position", type=float, default=100.0)
    args = ap.parse_args(argv)
    for name in ("runs", "runs_nofric"):
        raw = getattr(args, name)
        if raw is not None and not list(Path(raw).glob("seed*.pt")):
            ap.error(f"{name}: no checkpoints in {raw}")
    if args.logistic_model is not None and not Path(args.logistic_model).is_file():
        ap.error(f"logistic model missing: {args.logistic_model}")
    return args


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)

    print("=" * 74)
    print("EVALUATION ON HELD-OUT TEST SPLIT")
    print("=" * 74)
    print(split.summary())
    print(f"\n  test episodes: {len(split.test)}  (historical held-out partition)")

    # the existing logistic model from services/agent-py, if present
    lr_model = None
    if args.logistic_model is not None:
        with Path(args.logistic_model).open("rb") as f:
            lr_model = pickle.load(f)
        print(f"  loaded explicit logistic baseline from {args.logistic_model}")
    else:
        print("  no serialized baseline supplied; momentum fallback is labeled separately")

    # refit the same logistic method on THIS corpus's train split. the shipped
    # pickle keys on window_len, a constant in its training set, and was fit on
    # BTC prices in the tens of thousands rather than contract prices in [0,1].
    # both baselines are reported.
    lr_refit = fit_logistic_on_corpus(split.train)
    print(f"  logistic refit on train split: "
          f"{'ok' if lr_refit is not None else 'unavailable'}")

    cost = CostModel()
    test_env = BinaryMarketEnv(
        split.test,
        cost_model=cost,
        normalizer=split.normalizer,
        max_position=args.max_position,
        random_episode_order=False,
    )

    # ------------------------------------------------------------ baselines
    results: dict[str, Metrics] = {}
    raw: dict[str, np.ndarray] = {}
    episode_data = {"open_epoch": split.test.open_epoch}

    print("\nbaselines")
    for pol in default_baselines(logistic_model=lr_model, logistic_refit=lr_refit):
        r = run_policy(test_env, pol, seed=0)
        results[pol.name] = compute_metrics(
            r["pnl"], r["trades"], r["fees"], args.max_position
        )
        raw[pol.name] = r["pnl"]
        for key, values in r.items():
            episode_data[f"baseline_{pol.name}_{key}"] = values
        m = results[pol.name]
        print(f"  {pol.name:<16} pnl {m.mean_pnl:+8.4f}  sharpe {m.sharpe:+7.4f}  "
              f"trades {m.mean_trades:5.2f}  fees {m.mean_fees:6.3f}")

    # ------------------------------------------------------------------ ppo
    agents = load_agents(Path(args.runs))
    if not agents:
        print(f"\nno checkpoints in {args.runs}; run train_ppo.py first")
        raise SystemExit(1)

    print(f"\nppo across {len(agents)} seeds")
    ppo_rows, ppo_pnls = [], []
    ppo_trades, ppo_fees = [], []
    for i, agent in enumerate(agents):
        r = run_agent(test_env, agent)
        mm = compute_metrics(r["pnl"], r["trades"], r["fees"], args.max_position)
        ppo_rows.append(mm)
        ppo_pnls.append(r["pnl"])
        ppo_trades.append(r["trades"])
        ppo_fees.append(r["fees"])
        print(f"  seed {i}: pnl {mm.mean_pnl:+8.4f}  sharpe {mm.sharpe:+7.4f}  "
              f"trades {mm.mean_trades:5.2f}  hit {mm.hit_rate:.3f}")

    ppo_stack = np.stack(ppo_pnls)
    episode_data.update(ppo_pnl=ppo_stack, ppo_trades=np.stack(ppo_trades),
                        ppo_fees=np.stack(ppo_fees))
    ppo_mean_pnl = ppo_stack.mean(axis=0)
    raw["ppo"] = ppo_mean_pnl

    means = np.array([m.mean_pnl for m in ppo_rows])
    sharpes = np.array([m.sharpe for m in ppo_rows])
    print(f"\n  across seeds: pnl {means.mean():+.4f} +/- {means.std():.4f}   "
          f"sharpe {sharpes.mean():+.4f} +/- {sharpes.std():.4f}")

    # ------------------------------------------------- significance vs flat
    print("\nis any policy distinguishable from always-flat?")
    print("  (paired bootstrap, 10k resamples, on the same episodes)")
    flat = raw["always-flat"]
    sig = {}
    for name, pnl in raw.items():
        if name == "always-flat":
            continue
        p = paired_bootstrap_p_value(pnl, flat)
        sig[name] = p
        verdict = "distinguishable" if p < 0.05 else "not distinguishable"
        print(f"  {name:<16} diff {pnl.mean() - flat.mean():+8.4f}  p = {p:.4f}  {verdict}")

    # --------------------------------------------------------- ablation
    nofric_agents = load_agents(Path(args.runs_nofric)) if args.runs_nofric else []
    ablation = {}
    if nofric_agents:
        free_env = BinaryMarketEnv(
            split.test,
            cost_model=CostModel(enabled=False),
            normalizer=split.normalizer,
            max_position=args.max_position,
            random_episode_order=False,
        )
        free_raw = [run_agent(free_env, a) for a in nofric_agents]
        free_means = [r["pnl"].mean() for r in free_raw]
        for key in ("pnl", "trades", "fees"):
            episode_data[f"ppo_nofric_{key}"] = np.stack([r[key] for r in free_raw])
        ablation["ppo"] = float(np.mean(free_means))
        print(f"\nzero-cost ablation: ppo {np.mean(free_means):+.4f} "
              f"vs {means.mean():+.4f} with costs")

        for pol in default_baselines(logistic_model=lr_model, logistic_refit=lr_refit):
            r = run_policy(free_env, pol, seed=0)
            ablation[pol.name] = float(r["pnl"].mean())
            for key, values in r.items():
                episode_data[f"nofric_{pol.name}_{key}"] = values

    # ------------------------------------------------------------- figures
    print("\nfigures")
    best_baseline = min(
        (n for n in results if n != "always-flat"),
        key=lambda n: -results[n].mean_pnl,
    )
    plots.equity_curves(
        {
            "ppo (mean of seeds)": ppo_mean_pnl,
            "always-flat": raw["always-flat"],
            best_baseline: raw[best_baseline],
        },
        out / "equity_curves.png",
        band=(
            np.cumsum(ppo_stack, axis=1).min(axis=0),
            np.cumsum(ppo_stack, axis=1).max(axis=0),
        ),
        band_label="ppo seed range",
    )

    names = list(results.keys()) + ["ppo"]
    mus = [results[n].mean_pnl for n in results] + [float(means.mean())]
    errs = [0.0] * len(results) + [float(means.std())]
    plots.policy_comparison(names, mus, errs, out / "policy_comparison.png")

    logs = []
    for p in sorted(Path(args.runs).glob("seed*_log.json")):
        logs.append(json.loads(p.read_text()))
    if logs:
        plots.learning_curves(logs, out / "learning_curves.png")

    critic = critic_calibration(agents[0], test_env)
    cal = critic["bins"]
    episode_data.update(critic_predictions=critic["predictions"],
                        critic_realized_returns=critic["realized_returns"],
                        critic_stochastic_return_draws=critic["stochastic_return_draws"])
    if cal:
        err = sum(abs(r[1] - r[0]) * r[2] for r in cal) / sum(r[2] for r in cal)
        plots.value_calibration(
            cal,
            out / "value_calibration.png",
            subtitle=f"weighted mean |error| = {err:.4f}, {len(cal)} populated bins",
        )
        print(f"  critic calibration error: {err:.4f}")
    else:
        err = float("nan")

    if ablation:
        abl_names = [n for n in names if n in ablation]
        plots.cost_ablation(
            np.array([results[n].mean_pnl if n in results else float(means.mean())
                      for n in abl_names]),
            np.array([ablation[n] for n in abl_names]),
            out / "cost_ablation.png",
            labels=abl_names,
        )

    blocks = np.floor(split.test.open_epoch / 86400).astype(int)
    flat_groups = [np.zeros_like(row) for row in ppo_pnls]
    interval = cluster_bootstrap_ci(ppo_pnls, flat_groups, block_ids=blocks)
    block_test = cluster_bootstrap_p_value(ppo_pnls, flat_groups, block_ids=blocks)

    # -------------------------------------------------------------- results
    checkpoint_files = sorted(Path(args.runs).glob("seed*.pt"))
    if args.runs_nofric:
        checkpoint_files += sorted(Path(args.runs_nofric).glob("seed*.pt"))
    if args.logistic_model:
        checkpoint_files.append(Path(args.logistic_model))
    inputs = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in checkpoint_files}
    payload = {
        "model_input_sha256": inputs,
        "test_episodes": len(split.test),
        "split": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "purged": split.n_purged,
        },
        "baselines": {n: m.as_dict() for n, m in results.items()},
        "ppo_per_seed": [m.as_dict() for m in ppo_rows],
        "ppo_across_seeds": {
            "mean_pnl_mean": float(means.mean()),
            "mean_pnl_std": float(means.std()),
            "sharpe_mean": float(sharpes.mean()),
            "sharpe_std": float(sharpes.std()),
        },
        "paired_bootstrap_p_vs_flat": sig,
        "zero_cost_ablation": ablation,
        "critic_calibration_error": err,
        "critic_calibration": {"target": "stochastic on-policy return in dollars",
                               "repetitions": critic["repetitions"], "bins": cal},
        "ppo_seed_day_bootstrap": {"p_value": block_test, "ci95": list(interval),
                                   "n_seeds": len(ppo_pnls), "n_utc_days": len(np.unique(blocks)),
                                   "design": "crossed seeds and shared UTC-day episode blocks"},
    }
    np.savez_compressed(out / "evaluation_returns.npz", **episode_data)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
