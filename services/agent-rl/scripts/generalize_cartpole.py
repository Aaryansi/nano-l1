"""does the null test work outside the one market it was built on?

every other result in this project is measured on Kalshi binaries, which makes
them a case study. this runs the same test on CartPole: no trading, no
transaction costs, a well understood optimal policy, and four observation
features so all 2^4 = 16 coalitions can be enumerated. the shapley values are
EXACT, so nothing here can be blamed on sampling error.

three things are measured.

  A. a competence axis. checkpoints through training give a sequence of
     policies on an identical task, from useless to converged. the question is
     whether the test tracks competence rather than firing arbitrarily.

  B. two different nulls, compared directly.
       weight null       a randomly initialised network. this is the model
                         parameter randomization test of adebayo et al.
       environment null  an agent trained normally on a version of CartPole
                         whose observation channel carries no information.
     the second is this project's proposal and the first is the established
     one. if they disagree, that difference is the contribution; if they agree,
     the proposal is redundant and the report should say so.

  C. whether an undertrained agent's explanation is distinguishable from an
     explanation of nothing. that is the practitioner-facing question.

usage:
    python scripts/generalize_cartpole.py --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl import plots  # noqa: E402
from nano_rl.agents.networks import ActorCritic  # noqa: E402
from nano_rl.envs.gym_null import (  # noqa: E402
    BlindObservation,
    GymPPOConfig,
    evaluate_gym,
    exact_shapley_span,
    observation_moments,
    train_gym_ppo,
)
from nano_rl.explain.sanity import test_span_against_null  # noqa: E402

ENV_ID = "CartPole-v1"
FEATURES = ("cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity")


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def make_background(n: int = 512, seed: int = 0) -> np.ndarray:
    """reference observations, collected under a random policy."""
    env = gym.make(ENV_ID)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--null-steps", type=int, default=30_000)
    ap.add_argument("--attr-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bg = make_background(seed=args.seed)
    n_feat = len(FEATURES)

    # ------------------------------------------------------------- nulls
    banner("B. TWO NULLS: weight randomization vs environment randomization")

    print(f"  weight null: {args.n_null} randomly initialised networks")
    weight_spans = []
    for k in range(args.n_null):
        import torch

        torch.manual_seed(10_000 + k)
        net = ActorCritic(n_feat, 2, 64)
        sp, _ = exact_shapley_span(
            net, ENV_ID, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed + k
        )
        weight_spans.append(sp)
        print(f"    {k + 1}/{args.n_null}: span {sp:+8.3f}", flush=True)
    weight_spans = np.array(weight_spans)

    print(f"\n  environment null: {args.n_null} agents trained where the")
    print(f"  observation channel carries no information")
    mean, std = observation_moments(ENV_ID, seed=args.seed)
    env_spans = []
    for k in range(args.n_null):
        blind = BlindObservation(gym.make(ENV_ID), mean, std, seed=2000 + k)
        net, _ = train_gym_ppo(
            blind, GymPPOConfig(seed=2000 + k), total_steps=args.null_steps
        )
        blind.close()
        # attributed on the REAL environment: the agent has learned nothing
        # about the task, and we ask what its explanation looks like there.
        sp, _ = exact_shapley_span(
            net, ENV_ID, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed + k
        )
        env_spans.append(sp)
        print(f"    {k + 1}/{args.n_null}: span {sp:+8.3f}", flush=True)
    env_spans = np.array(env_spans)

    print(f"\n  weight null      : {weight_spans.mean():+8.3f} "
          f"+/- {weight_spans.std(ddof=1):.3f}")
    print(f"  environment null : {env_spans.mean():+8.3f} "
          f"+/- {env_spans.std(ddof=1):.3f}")

    # ----------------------------------------------------- competence axis
    banner("A. COMPETENCE AXIS: checkpoints through training")

    fractions = (0.05, 0.15, 0.35, 0.70, 1.0)
    env = gym.make(ENV_ID)
    final, checkpoints = train_gym_ppo(
        env, GymPPOConfig(seed=args.seed), total_steps=args.steps,
        checkpoint_fractions=fractions,
    )
    env.close()

    print(f"  {'progress':>9} {'return':>9} {'span':>9} "
          f"{'z (env null)':>14} {'z (weight null)':>16} {'detected':>10}")

    rows = []
    for frac, net in checkpoints:
        ret = evaluate_gym(net, ENV_ID, n_episodes=25, seed=args.seed)
        span, values = exact_shapley_span(
            net, ENV_ID, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed
        )
        r_env = test_span_against_null(span, env_spans)
        r_wt = test_span_against_null(span, weight_spans)

        rows.append(
            {
                "fraction": frac,
                "return": ret,
                "span": span,
                "shapley": values.tolist(),
                "z_env_null": r_env.z_score,
                "z_weight_null": r_wt.z_score,
                "detected_env": bool(r_env.passes),
                "detected_weight": bool(r_wt.passes),
            }
        )
        print(
            f"  {frac:>8.0%} {ret:>9.1f} {span:>9.2f} {r_env.z_score:>+14.2f} "
            f"{r_wt.z_score:>+16.2f} {'YES' if r_env.passes else 'no':>10}",
            flush=True,
        )

    # ------------------------------------------------------------- verdict
    banner("WHAT THIS SHOWS")

    detected = [r for r in rows if r["detected_env"]]
    if detected:
        first = min(detected, key=lambda r: r["return"])
        print(f"  the test first fires at return {first['return']:.0f} / 500 "
              f"({first['fraction']:.0%} of training).")
        undet = [r for r in rows if not r["detected_env"]]
        if undet:
            worst = max(undet, key=lambda r: r["return"])
            print(f"  agents up to return {worst['return']:.0f} are NOT")
            print(f"  distinguishable from agents that learned nothing.")
    else:
        print("  the test never fired, which would mean it has no power here")

    agree = sum(1 for r in rows if r["detected_env"] == r["detected_weight"])
    print(f"\n  the two nulls agree on {agree}/{len(rows)} checkpoints.")
    if agree == len(rows):
        print("  they are interchangeable on this task, so the environment null")
        print("  is not adding anything here and the report should say so.")
    else:
        print("  they disagree, so the choice of null changes the verdict.")

    print(f"\n  shapley values are EXACT here (all {2**n_feat} coalitions")
    print(f"  enumerated), so none of this is sampling error.")

    plots.competence_curve(
        [r["return"] for r in rows],
        [r["z_env_null"] for r in rows],
        [r["detected_env"] for r in rows],
        [r["fraction"] for r in rows],
        out / "cartpole_competence.png",
        subtitle=f"CartPole-v1, exact shapley over {2**n_feat} coalitions; "
                 f"null from {len(env_spans)} agents with an uninformative "
                 f"observation channel",
    )

    (out / "cartpole.json").write_text(json.dumps(
        {
            "weight_null_spans": weight_spans.tolist(),
            "env_null_spans": env_spans.tolist(),
            "checkpoints": rows,
            "features": list(FEATURES),
        },
        indent=2,
    ))
    print(f"\nwrote {out}/cartpole.json")


if __name__ == "__main__":
    main()
