"""Describe reference widths across fixed synthetic episode horizons.

All policies share independent evaluation episodes. Each trained policy retains
its training normalizer when evaluating both real and replacement observations.
Width and log-log fits are descriptive; they do not measure detection power or
establish a mechanism unique to reinforcement learning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl import plots
from nano_rl.agents.networks import ActorCritic
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.features import N_FEATURES, fit_normalizer
from nano_rl.env.synthetic import make_null_corpus, synthetic_train_eval
from nano_rl.explain.rollout import VectorizedRollout, build_background, masked_span


def span_of(net, batch, norm, background_seed, masking_seed):
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.)
    background = build_background(roll, n_samples=192, seed=background_seed)
    return masked_span(SimpleNamespace(net=net), roll, background, masking_seed,
                       return_samples=True)


def measure_horizon(n_steps, n_null, updates, n_episodes, seed, *, raw=None):
    pair = synthetic_train_eval(make_null_corpus, n_episodes=n_episodes,
                                n_steps=n_steps, seed=seed)
    batch, norm = pair.evaluation, fit_normalizer(pair.train)
    weight, env_null = [], []
    keys = {"weight": [], "env": []}
    normalizers = []

    def retain(arm, k, record):
        key = f"h{n_steps}_{arm}_{k}"
        keys[arm].append(key)
        if raw is not None:
            raw[f"{key}_full"] = record["full"]
            raw[f"{key}_masked"] = record["masked"]
            raw[f"{key}_open_epoch"] = batch.open_epoch
        return record["span"]

    for k in range(n_null):
        torch.manual_seed(10000 + k)
        net = ActorCritic(N_FEATURES, 3, 64)
        weight.append(retain("weight", k, span_of(net, batch, norm, seed, seed + k)))
    for k in range(n_null):
        train_seed = 5000 + seed + k
        training = make_null_corpus(n_episodes=n_episodes, n_steps=n_steps, seed=train_seed)
        training_norm = fit_normalizer(training)
        env = BinaryMarketEnv(training, max_position=100., normalizer=training_norm)
        env.reset(seed=seed + k)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed + k))
        agent.train(env, n_updates=updates, verbose=False)
        env_null.append(retain("env", k, span_of(agent.net, batch, training_norm, seed, seed + k)))
        normalizers.append(training_norm.state_dict())
    w, v = np.asarray(weight), np.asarray(env_null)
    ws, vs = float(w.std(ddof=1)), float(v.std(ddof=1))
    return {"n_steps": n_steps, "clock_feature_deadline_seconds": batch.duration_s,
            "weight_std": ws, "env_std": vs,
            "weight_spans": weight, "env_spans": env_null,
            "ratio": ws / vs if vs > 1e-12 else None,
            "endpoint_keys": keys, "shared_data": pair.metadata(),
            "weight_normalizer": norm.state_dict(), "reference_normalizers": normalizers,
            "reference_train_seeds": list(range(5000 + seed, 5000 + seed + n_null)),
            "reference_policy_seeds": list(range(seed, seed + n_null)),
            "weight_initialization_seeds": list(range(10000, 10000 + n_null)),
            "masking_seeds": list(range(seed, seed + n_null)), "background_seed": seed}


def fit_power_law(x, y):
    """Descriptive least-squares slope and fit quality on positive points only."""
    keep = (x > 0) & (y > 0)
    if keep.sum() < 3:
        return None, None
    lx, ly = np.log(x[keep]), np.log(y[keep])
    alpha, intercept = np.polyfit(lx, ly, 1)
    residual = float(((ly - alpha * lx - intercept) ** 2).sum())
    total = float(((ly - ly.mean()) ** 2).sum())
    return float(alpha), 1. - residual / total if total > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 7, 14, 28, 56])
    ap.add_argument("--n-null", type=int, default=16)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if (args.n_null < 2 or args.updates < 1 or args.episodes < 2
            or min(args.horizons) < 2 or len(set(args.horizons)) != len(args.horizons)):
        raise SystemExit("require at least two references/episodes, positive updates and distinct horizons of at least two steps")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = {}
    rows = [measure_horizon(h, args.n_null, args.updates, args.episodes, args.seed, raw=raw)
            for h in args.horizons]
    for row in rows:
        ratio = "undefined" if row["ratio"] is None else f"{row['ratio']:.3f}"
        print(f"horizon {row['n_steps']}: weight SD {row['weight_std']:.4f}, "
              f"trained reference SD {row['env_std']:.4f}, ratio {ratio}", flush=True)
    x = np.array([row["n_steps"] for row in rows], dtype=float)
    w = np.array([row["weight_std"] for row in rows])
    v = np.array([row["env_std"] for row in rows])
    aw, rw = fit_power_law(x, w)
    av, rv = fit_power_law(x, v)
    scope = "descriptive fits only; clock inputs and trading opportunities also vary; widths do not measure power or isolate a horizon mechanism"
    print(scope)
    plots.horizon_scaling(x, w, v, args.out / "horizon_scaling.png",
                          alpha_weight=aw, alpha_env=av,
                          subtitle=f"{args.n_null} references per horizon; descriptive log-log fits")
    np.savez_compressed(args.out / "horizon_endpoint_returns.npz", **raw)
    (args.out / "horizon_scaling.json").write_text(json.dumps({
        "rows": rows, "alpha_weight": aw, "r2_weight": rw, "alpha_env": av, "r2_env": rv,
        "verdict": scope, "fit_scope": "positive standard deviations only; no post-selection refit",
        "fit_horizons": {"weight": x[w > 0].tolist(), "env": x[v > 0].tolist()},
        "zero_width_horizons": [row["n_steps"] for row in rows if row["env_std"] <= 1e-12],
        "evaluation_episodes": args.episodes, "training_updates": args.updates,
        "step_seconds": 60, "clock_feature_deadline": "inferred per horizon from the decision grid",
        "background_samples": 192, "endpoint_returns_file": "horizon_endpoint_returns.npz",
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
