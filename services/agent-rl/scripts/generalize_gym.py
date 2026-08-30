"""does the null-test result hold across environments, or only on CartPole?

supersedes scripts/generalize_cartpole.py, which ran one task. the finding
being checked is not about trading or about any particular domain: it is that
the ESTABLISHED null construction, randomizing network weights, produces a
reference distribution so wide in reinforcement learning that a test built on
it has no power, while randomizing the environment's information content does
not.

that claim rests on a comparison between two nulls, so it needs to hold on more
than one task to be a claim about RL rather than about CartPole.

both environments here have small enough observation spaces to enumerate every
coalition, so the per-feature Shapley values reported for the final agent are
exact. the test statistic itself is the span v(all) - v(none), which by the
efficiency axiom needs only the two endpoint evaluations rather than all 2^n.

usage:
    python scripts/generalize_gym.py --envs CartPole-v1 Acrobot-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl import plots  # noqa: E402
from nano_rl.agents.networks import ActorCritic  # noqa: E402
from nano_rl.envs.gym_null import (  # noqa: E402
    BlindObservation,
    GymPPOConfig,
    attribution_span_fast,
    evaluate_gym,
    exact_shapley_span,
    make_env,
    observation_moments,
    train_gym_ppo,
)
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def make_background(env_id: str, n: int = 512, seed: int = 0) -> np.ndarray:
    env = make_env(env_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    while len(rows) < n:
        rows.append(np.asarray(obs, dtype=np.float32))
        obs, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.asarray(rows, dtype=np.float32)


def run_env(env_id: str, args) -> dict:
    probe = make_env(env_id)
    n_feat = int(probe.observation_space.shape[0])
    n_act = int(probe.action_space.n)
    probe.close()

    banner(f"{env_id}   {n_feat} features, {n_act} actions, "
           f"{2 ** n_feat} coalitions")

    bg = make_background(env_id, seed=args.seed)

    # ---- null 1: randomised weights (the established construction)
    #
    # each net's unmasked return is recorded alongside its span, to test the
    # conjecture left open by the horizon experiment: that the weight null is
    # wide because a randomly initialised network is sometimes an accidentally
    # competent policy. both quantities are in return units, so they can be
    # compared directly rather than through a normalisation that would have to
    # be argued for.
    weight_spans, weight_returns = [], []
    for k in range(args.n_null):
        torch.manual_seed(10_000 + k)
        net = ActorCritic(n_feat, n_act, 64)
        weight_spans.append(
            attribution_span_fast(
                net, env_id, bg, n_feat,
                n_episodes=args.attr_episodes, seed=args.seed + k,
            )
        )
        weight_returns.append(
            evaluate_gym(net, env_id, n_episodes=args.attr_episodes,
                         seed=args.seed + k)
        )
    weight_spans = np.array(weight_spans)
    weight_returns = np.array(weight_returns)

    # ---- null 2: uninformative observation channel (this work)
    mean, std = observation_moments(env_id, seed=args.seed)
    env_spans = []
    for k in range(args.n_null):
        blind = BlindObservation(make_env(env_id), mean, std, seed=2000 + k)
        net, _ = train_gym_ppo(
            blind, GymPPOConfig(seed=2000 + k), total_steps=args.null_steps
        )
        blind.close()
        env_spans.append(
            attribution_span_fast(
                net, env_id, bg, n_feat,
                n_episodes=args.attr_episodes, seed=args.seed + k,
            )
        )
    env_spans = np.array(env_spans)

    print(f"  random-init return: {weight_returns.mean():>+9.3f} "
          f"+/- {weight_returns.std(ddof=1):>8.3f}")
    print(f"  weight null      : {weight_spans.mean():>+9.3f} "
          f"+/- {weight_spans.std(ddof=1):>8.3f}")
    print(f"  environment null : {env_spans.mean():>+9.3f} "
          f"+/- {env_spans.std(ddof=1):>8.3f}")
    ratio = weight_spans.std(ddof=1) / max(env_spans.std(ddof=1), 1e-9)
    print(f"  the weight null is {ratio:.0f}x wider\n")

    # ---- the trained agent, checkpointed
    fractions = (0.1, 0.25, 0.5, 1.0)
    env = make_env(env_id)
    final, checkpoints = train_gym_ppo(
        env, GymPPOConfig(seed=args.seed), total_steps=args.steps,
        checkpoint_fractions=fractions,
    )
    env.close()

    print(f"  {'progress':>9} {'return':>10} {'span':>10} "
          f"{'z (env)':>10} {'z (weight)':>12} {'agree?':>8}")

    rows = []
    for frac, net in checkpoints:
        ret = evaluate_gym(net, env_id, n_episodes=25, seed=args.seed)
        span = attribution_span_fast(
            net, env_id, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed
        )
        r_env = test_span_against_null(span, env_spans)
        r_wt = test_span_against_null(span, weight_spans)
        same = r_env.passes == r_wt.passes
        rows.append(
            {
                "fraction": frac, "return": ret, "span": span,
                "z_env": r_env.z_score, "z_weight": r_wt.z_score,
                "detected_env": bool(r_env.passes),
                "detected_weight": bool(r_wt.passes),
            }
        )
        print(f"  {frac:>8.0%} {ret:>10.1f} {span:>10.2f} {r_env.z_score:>+10.2f} "
              f"{r_wt.z_score:>+12.2f} {'yes' if same else 'NO':>8}", flush=True)

    # exact per-feature values for the converged agent only
    _, exact_vals = exact_shapley_span(
        final, env_id, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed
    )
    agree = sum(1 for r in rows if r["detected_env"] == r["detected_weight"])
    print(f"\n  the two nulls agree on {agree}/{len(rows)} checkpoints")

    return {
        "env_id": env_id,
        "n_features": n_feat,
        "weight_null": {"mean": float(weight_spans.mean()),
                        "std": float(weight_spans.std(ddof=1)),
                        "spans": weight_spans.tolist()},
        "random_init_return": {"mean": float(weight_returns.mean()),
                               "std": float(weight_returns.std(ddof=1)),
                               "returns": weight_returns.tolist()},
        "env_null": {"mean": float(env_spans.mean()),
                     "std": float(env_spans.std(ddof=1)),
                     "spans": env_spans.tolist()},
        "width_ratio": float(ratio),
        "checkpoints": rows,
        "exact_shapley_final": exact_vals.tolist(),
        "nulls_agree": agree,
        "n_checkpoints": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["CartPole-v1", "Acrobot-v1"])
    ap.add_argument("--out", default="reports")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--null-steps", type=int, default=30_000)
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--attr-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = [run_env(e, args) for e in args.envs]

    banner("ACROSS ENVIRONMENTS")
    print(f"  {'environment':<16}{'weight null sd':>16}{'env null sd':>14}"
          f"{'ratio':>14}{'nulls agree':>12}")
    for r in results:
        ratio = r["width_ratio"]
        # a degenerate null has zero variance, so the ratio is meaningless
        # rather than merely large; say so instead of printing 1e11.
        ratio_s = "degenerate" if ratio > 1e6 else f"{ratio:.0f}x"
        print(f"  {r['env_id']:<16}{r['weight_null']['std']:>16.2f}"
              f"{r['env_null']['std']:>14.4f}{ratio_s:>14}"
              f"{r['nulls_agree']:>10}/{r['n_checkpoints']}")

    total_pairs = sum(r["n_checkpoints"] for r in results)
    total_agree = sum(r["nulls_agree"] for r in results)
    print(f"\n  overall the two nulls agree on {total_agree}/{total_pairs} "
          f"checkpoints across {len(results)} environments")
    if total_agree == 0:
        print("  they never agree, so the choice of null decides every verdict")

    # ---- the conjecture the horizon experiment left open
    banner("IS THE WEIGHT NULL WIDE BECAUSE RANDOM NETS ARE SOMETIMES COMPETENT?")
    print("  conjectured mechanism: a randomly initialised net is occasionally")
    print("  an accidentally decent policy, so masking its inputs costs a lot")
    print("  on those draws and nothing on the rest. if so, the spread of")
    print("  random-init RETURNS should track the spread of their spans.")
    print()
    print(f"  {'environment':<18}{'random return sd':>18}{'weight null sd':>17}"
          f"{'ratio':>9}")
    rr, ws = [], []
    for r in results:
        a = r["random_init_return"]["std"]
        b = r["weight_null"]["std"]
        rr.append(a)
        ws.append(b)
        print(f"  {r['env_id']:<18}{a:>18.2f}{b:>17.2f}{b / max(a, 1e-9):>9.2f}")

    conj: dict = {"n_environments": len(results)}
    if len(results) >= 3:
        c = float(np.corrcoef(rr, ws)[0, 1])
        conj["return_sd_vs_null_sd_corr"] = c
        print(f"\n  correlation across {len(results)} environments: {c:+.3f}")
        # with a handful of environments this is suggestive at best, and the
        # paper must not report it as though it were an estimate.
        print("  n is small; this is consistent-with, not evidence-for.")
        conj["supported"] = bool(c > 0.8)
    else:
        print("\n  too few environments to correlate anything")
    conj["random_return_sd"] = list(map(float, rr))
    conj["weight_null_sd"] = list(map(float, ws))
    conj["env_ids"] = [r["env_id"] for r in results]

    plots.null_comparison(
        {r["env_id"]: (r["weight_null"]["spans"], r["env_null"]["spans"])
         for r in results},
        out / "null_comparison.png",
        subtitle="the established weight-randomization null against an "
                 "environment-level one",
    )

    (out / "generalize_gym.json").write_text(json.dumps(results, indent=2))
    # kept in its own file: generalize_gym.json is a bare list consumed
    # positionally elsewhere, and wrapping it to add a key would break readers
    # for no gain.
    (out / "null_width_conjecture.json").write_text(json.dumps(conj, indent=2))
    print(f"\nwrote {out}/generalize_gym.json")
    print(f"wrote {out}/null_width_conjecture.json")


if __name__ == "__main__":
    main()
