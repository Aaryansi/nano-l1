"""Compare objectives on the same real episodes with matched training budgets.

Both policies are trained from scratch at each requested budget. State
transitions, observations, normalization, evaluation episodes, and initialization
seeds are shared. Rewards differ. Per-seed standardized-span contrasts are
descriptive; differing arm-wise verdicts do not establish an objective effect.
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
from nano_rl.env.features import N_FEATURES  # noqa: E402
from nano_rl.env.prediction import (  # noqa: E402
    BlindEnv,
    PredictionEnv,
    PredictionRollout,
    observation_moments,
)
from nano_rl.explain.rollout import (  # noqa: E402
    VectorizedRollout,
    build_background,
    masked_span,
)
from nano_rl.explain.sanity import (  # noqa: E402
    check_resolving_power,
    short_verdict,
    test_span_against_null,
)
from run_manifest import digest  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train(env, updates: int, seed: int) -> PPOAgent:
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def run_task(name, make_env, make_roll, split, updates, seeds, n_null, seed):
    """Train both objectives from scratch at identical explicit budgets."""
    banner(f"{name.upper()}: {updates} updates, {seeds} independent agents")
    roll = make_roll(split.test)
    background = build_background(
        VectorizedRollout(split.test, normalizer=split.normalizer, max_position=100.),
        n_samples=192, seed=seed)
    mean, sd = observation_moments(make_env(split.train), seed=seed)
    nulls, reference_full, reference_masked = [], [], []
    for k in range(n_null):
        policy_seed = 10000 + seed + k
        blind = BlindEnv(make_env(split.train), mean, sd, seed=3000 + k)
        agent = train(blind, updates, policy_seed)
        endpoints = masked_span(agent, roll, background, policy_seed, return_samples=True)
        nulls.append(endpoints["span"])
        reference_full.append(endpoints["full"])
        reference_masked.append(endpoints["masked"])
        print(f"  reference {k + 1}/{n_null}: {nulls[-1]:+.3f}", flush=True)
    rows, raw_returns, masked_returns = [], [], []
    for k in range(seeds):
        agent = train(make_env(split.train), updates, seed + k)
        endpoints = masked_span(agent, roll, background, seed, return_samples=True)
        span, returns = endpoints["span"], endpoints["full"]
        comparison = test_span_against_null(span, nulls)
        result = comparison.as_dict()
        if result["null_std"] <= 1e-12:
            result["z_score"] = result["p_normal"] = None
        row = {"seed": seed + k, "return": float(returns.mean()), "span": span,
               "masking_seed": seed, "result": result, "fires": comparison.passes,
               "verdict": short_verdict(comparison)}
        rows.append(row)
        raw_returns.append(returns)
        masked_returns.append(endpoints["masked"])
        print(f"  seed {seed + k}: {comparison.summary()}", flush=True)
    primary = rows[0]
    return {"task": name, "updates": updates, "n_seeds": seeds,
            "return": primary["return"], "span": primary["span"],
            "result": primary["result"], "fires": primary["fires"],
            "verdict": primary["verdict"], "per_seed": rows,
            "return_mean_across_seeds": float(np.mean([r["return"] for r in rows])),
            "span_mean_across_seeds": float(np.mean([r["span"] for r in rows])),
            "null_spans": nulls, "null_mean": float(np.mean(nulls)),
            "null_std": float(np.std(nulls, ddof=1)),
            "reference_policy_seeds": [10000 + seed + k for k in range(n_null)],
            "reference_masking_seeds": [10000 + seed + k for k in range(n_null)],
            "reference_blinding_seeds": [3000 + k for k in range(n_null)],
            "background_seed": seed, "background_samples": 192,
            "observation_moment_seed": seed}, {
                "returns": np.stack(raw_returns), "masked_returns": np.stack(masked_returns),
                "reference_returns": np.stack(reference_full),
                "reference_masked_returns": np.stack(reference_masked)}


def direct_contrast(prediction, trading):
    """Describe paired standardized-span differences without a significance claim."""
    pairs = []
    for a, b in zip(prediction["per_seed"], trading["per_seed"], strict=True):
        za, zb = a["result"]["z_score"], b["result"]["z_score"]
        defined = (a["result"]["null_std"] > 1e-12 and b["result"]["null_std"] > 1e-12
                   and np.isfinite(za) and np.isfinite(zb))
        difference = float(za - zb) if defined else None
        pairs.append({"seed": a["seed"], "difference": difference})
    values = [r["difference"] for r in pairs if r["difference"] is not None]
    return {"estimand": "prediction minus trading standardized span, conditional on estimated references",
            "per_seed": pairs, "n_defined": len(values),
            "mean_difference": float(np.mean(values)) if values else None,
            "interpretation": "descriptive contrast; differing arm-wise verdicts are not a significant objective effect"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=24)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--budgets", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    budgets = args.budgets or [args.updates]
    if (args.seeds < 1 or args.n_null < 2 or args.seed < 0 or any(n < 1 for n in budgets)
            or len(set(budgets)) != len(budgets)):
        raise SystemExit("require positive seeds/unique budgets, at least two references and a nonnegative seed")
    check_resolving_power(args.n_null, what="the matched objective-control reference")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("positive_control.json", "positive_control_returns.npz")):
        raise SystemExit("refusing existing objective-control evidence; choose a fresh output")
    corpus = Path(args.corpus)
    corpus_hash = digest(corpus)
    split = walk_forward_split(EpisodeBatch.load(str(corpus)))
    if digest(corpus) != corpus_hash:
        raise SystemExit("objective corpus changed while loading")
    norm = split.normalizer
    tasks = [
        ("prediction", lambda b: PredictionEnv(b, normalizer=norm, max_position=100.),
         lambda b: PredictionRollout(b, normalizer=norm, max_position=100.)),
        ("trading", lambda b: BinaryMarketEnv(b, normalizer=norm, max_position=100.),
         lambda b: VectorizedRollout(b, normalizer=norm, max_position=100.)),
    ]
    sweep, raw = [], []
    for updates in budgets:
        rows, raw_budget = [], []
        for name, make_env, make_roll in tasks:
            row, returns = run_task(name, make_env, make_roll, split, updates,
                                    args.seeds, args.n_null, args.seed)
            rows.append(row)
            raw_budget.append(returns)
        undecided = [r["task"] for r in rows if r["verdict"] == "unresolved"]
        separated = None if undecided else rows[0]["fires"] and not rows[1]["fires"]
        sweep.append({"updates": updates, "tasks": rows,
                      "contrast": direct_contrast(*rows), "separated": separated,
                      "undecided_tasks": undecided})
        raw.append(raw_budget)
    if digest(corpus) != corpus_hash:
        raise SystemExit("objective corpus changed during execution; refusing to save evidence")
    archive = out / "positive_control_returns.npz"
    arrays = {key: np.asarray([[task[key] for task in budget] for budget in raw]) for key in raw[0][0]}
    np.savez_compressed(archive, **arrays, budgets=np.asarray(budgets),
                        task_names=np.asarray([task[0] for task in tasks]), open_epoch=split.test.open_epoch,
                        agent_seeds=np.arange(args.seeds) + args.seed,
                        observed_masking_seeds=np.full(args.seeds, args.seed),
                        reference_policy_seeds=np.arange(args.n_null) + 10000 + args.seed,
                        reference_masking_seeds=np.arange(args.n_null) + 10000 + args.seed,
                        reference_blinding_seeds=np.arange(args.n_null) + 3000)
    payload = dict(sweep[0], budget_sweep=sweep, n_train=len(split.train), n_test=len(split.test),
                   seed=args.seed, n_null=args.n_null, corpus_sha256=corpus_hash,
                   normalizer=norm.state_dict(), endpoint_schema_version=2,
                   endpoint_returns_file=archive.name, endpoint_returns_sha256=digest(archive),
                   design="matched objectives and budgets; freshly trained policies")
    (out / "positive_control.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    from verify_objective_control import verify
    verify(out, Path(args.corpus))
    print(f"wrote {out / 'positive_control.json'}")


if __name__ == "__main__":
    main()
