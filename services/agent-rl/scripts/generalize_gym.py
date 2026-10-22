"""Compare reference constructions on six control-task configurations.

Report checkpoint and independent-seed decisions alongside competence gates,
observation dependence and reference resolution. Failed learning or a collapsed
legacy reference is not evidence of low detection power. Reference widths are
in each task's reward units and do not by themselves measure test power.
Exact feature coalitions where feasible and endpoint spans address the chosen
intervention, not general feature-level fidelity or a universal RL null.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
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
    observation_pool,
    train_gym_ppo,
)
from nano_rl.explain.sanity import (  # noqa: E402
    check_resolving_power,
    short_verdict,
    summarize_verdicts,
    test_span_against_null,
)


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


# Declared convergence gates for these small control tasks, in native reward units.
COMPETENCE_RETURN = {"CartPole-v1": 475., "Acrobot-v1": -100.,
                     "MountainCar-v0": -110., "Pendulum-v1": -300.,
                     "FrozenLake-v1": .8, "CliffWalking-v0": -20.}


def competence_gate(env_id, episode_return, span):
    threshold = COMPETENCE_RETURN.get(env_id)
    learned = threshold is not None and episode_return >= threshold
    dependent = abs(span) > 1e-6
    return {"return_threshold": threshold, "meets_return_threshold": learned,
            "observation_dependent": dependent,
            "eligible_for_sensitivity_claim": learned and dependent}


def training_config(seed, backend):
    if backend == "sb3":
        return GymPPOConfig(seed=seed, backend=backend, update_epochs=10,
                            minibatch_size=64, entropy_coef=0., n_envs=4)
    return GymPPOConfig(seed=seed, backend=backend)


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
    weight_spans, weight_returns, weight_rollouts = [], [], []
    for k in range(args.n_null):
        torch.manual_seed(10_000 + k)
        net = ActorCritic(n_feat, n_act, 64)
        raw = attribution_span_fast(net, env_id, bg, n_feat,
            n_episodes=args.attr_episodes, seed=args.seed + k, return_samples=True)
        raw["initialization_seed"] = 10_000 + k
        weight_rollouts.append(raw)
        weight_spans.append(raw["span"])
        weight_returns.append(float(np.mean(raw["full"])))
    weight_spans = np.array(weight_spans)
    weight_returns = np.array(weight_returns)

    # ---- null 2: uninformative observation channel (this work)
    mean, std = observation_moments(env_id, seed=args.seed)
    # blinding must remove the information without leaving the observation
    # manifold. independent per-coordinate gaussians do leave it: on Pendulum
    # the observation is (cos, sin, thetadot) and cos^2+sin^2 is identically
    # one, which moment-matched draws satisfy for well under 1% of samples.
    # resampling whole observations keeps every within-observation constraint
    # exact while still carrying no information about the current state.
    pool = observation_pool(env_id, seed=args.seed) if args.blind == "resample" else None
    env_spans, env_rollouts = [], []
    for k in range(args.n_null):
        blind = BlindObservation(make_env(env_id), mean, std, seed=2000 + k,
                                 pool=pool)
        net, _ = train_gym_ppo(
            blind, training_config(2000 + k, args.trainer), total_steps=args.null_steps
        )
        blind.close()
        raw = attribution_span_fast(net, env_id, bg, n_feat,
            n_episodes=args.attr_episodes, seed=args.seed + k, return_samples=True)
        raw["training_seed"] = 2000 + k
        env_rollouts.append(raw)
        env_spans.append(raw["span"])
    env_spans = np.array(env_spans)

    print(f"  random-init return: {weight_returns.mean():>+9.3f} "
          f"+/- {weight_returns.std(ddof=1):>8.3f}")
    print(f"  weight null      : {weight_spans.mean():>+9.3f} "
          f"+/- {weight_spans.std(ddof=1):>8.3f}")
    print(f"  environment null : {env_spans.mean():>+9.3f} "
          f"+/- {env_spans.std(ddof=1):>8.3f}")
    ratio = (float(weight_spans.std(ddof=1) / env_spans.std(ddof=1))
             if env_spans.std(ddof=1) > 1e-12 else None)
    print(f"  weight/environment reference width ratio: {ratio}\n")

    # ---- the trained agent
    #
    # two modes, and the difference matters for what can be claimed. snapshotting
    # one training run at four fractions gives four CORRELATED observations of a
    # single trajectory: they share an initialisation, a data order and every
    # gradient step up to the earliest snapshot. that supports a statement about
    # how the verdict moves as one agent learns, and not a statement about how
    # the two constructions behave across agents. for the latter the replicate
    # has to be the training run, so --agent-seeds trains that many independent
    # agents to full length and reports the spread across them.
    replicate_mode = args.agent_seeds > 1
    if replicate_mode:
        units = []
        for k in range(args.agent_seeds):
            env = make_env(env_id)
            net, _ = train_gym_ppo(
                env, training_config(args.seed + k, args.trainer), total_steps=args.steps
            )
            env.close()
            units.append((args.seed + k, net))
            print(f"  trained independent agent {k + 1}/{args.agent_seeds}",
                  flush=True)
        final = units[-1][1]
        unit_key, unit_name = "seed", "seed"
    else:
        fractions = (0.1, 0.25, 0.5, 1.0)
        env = make_env(env_id)
        final, units = train_gym_ppo(
            env, training_config(args.seed, args.trainer), total_steps=args.steps,
            checkpoint_fractions=fractions,
        )
        env.close()
        unit_key, unit_name = "fraction", "progress"

    print(f"  {unit_name:>9} {'return':>10} {'span':>10} "
          f"{'z (env)':>10} {'z (weight)':>12} {'agree?':>8}")

    rows = []
    for frac, net in units:
        return_samples = evaluate_gym(net, env_id, n_episodes=25, seed=args.seed, return_samples=True)
        ret = float(np.mean(return_samples))
        span_rollouts = attribution_span_fast(
            net, env_id, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed,
            return_samples=True
        )
        span = span_rollouts["span"]
        r_env = test_span_against_null(span, env_spans)
        r_wt = test_span_against_null(span, weight_spans)
        same = summarize_verdicts([r_env, r_wt])["agreement"]
        rows.append(
            {
                unit_key: frac, "return": ret, "span": span,
                "return_rollouts": return_samples.tolist(),
                "return_episode_seeds": list(range(args.seed, args.seed + 25)),
                "span_rollouts": span_rollouts,
                "competence": competence_gate(env_id, ret, span),
                "z_env": r_env.z_score, "z_weight": r_wt.z_score,
                "detected_env": bool(r_env.passes),
                "detected_weight": bool(r_wt.passes),
                "verdict_env": short_verdict(r_env),
                "verdict_weight": short_verdict(r_wt),
                "reference_agreement": same,
            }
        )
        shown = f"{frac:>8.0%}" if not replicate_mode else f"{frac:>9d}"
        print(f"  {shown} {ret:>10.1f} {span:>10.2f} {r_env.z_score:>+10.2f} "
              f"{r_wt.z_score:>+12.2f} "
              f"{('unresolved' if same is None else 'yes' if same else 'NO'):>10}", flush=True)

    # exact per-feature values for the converged agent only
    _, exact_vals = exact_shapley_span(
        final, env_id, bg, n_feat, n_episodes=args.attr_episodes, seed=args.seed
    )
    resolved_count = sum(r["reference_agreement"] is not None for r in rows)
    agree = sum(r["reference_agreement"] is True for r in rows)
    noun = "independently trained agents" if replicate_mode else "checkpoints"
    print(f"\n  resolved reference decisions agree on {agree}/{resolved_count} {noun}; "
          f"{len(rows) - resolved_count} comparisons unresolved")
    if replicate_mode:
        rets = np.array([r["return"] for r in rows], dtype=float)
        spns = np.array([r["span"] for r in rows], dtype=float)
        vs = {r["verdict_env"] for r in rows}
        print(f"  across seeds: return {rets.mean():+.1f} +/- {rets.std(ddof=1):.1f}, "
              f"span {spns.mean():+.2f} +/- {spns.std(ddof=1):.2f}")
        print(f"  environment-null verdict is "
              f"{'the same on every seed' if len(vs) == 1 else f'NOT stable: {sorted(vs)}'}")

    return {
        "env_id": env_id,
        "execution": {"workers": getattr(args, "worker_count", 1),
                      "start_method": "spawn" if getattr(args, "worker_count", 1) > 1 else "single process",
                      "torch_threads": torch.get_num_threads()},
        "background_seed": args.seed,
        "attribution_episodes": args.attr_episodes,
        "competence_episodes": 25,
        "trainer": args.trainer,
        "training_config": asdict(training_config(args.seed, args.trainer)),
        "n_eligible_agents": sum(r["competence"]["eligible_for_sensitivity_claim"] for r in rows),
        "competence_policy": "failed or observation-independent agents do not establish detection sensitivity",
        "n_features": n_feat,
        "weight_null": {"mean": float(weight_spans.mean()),
                        "std": float(weight_spans.std(ddof=1)),
                        "spans": weight_spans.tolist(), "rollouts": weight_rollouts},
        "random_init_return": {"mean": float(weight_returns.mean()),
                               "std": float(weight_returns.std(ddof=1)),
                               "returns": weight_returns.tolist()},
        "env_null": {"mean": float(env_spans.mean()),
                     "std": float(env_spans.std(ddof=1)),
                     "spans": env_spans.tolist(), "rollouts": env_rollouts},
        "width_ratio": ratio,
        "checkpoints": rows,
        "exact_shapley_final": exact_vals.tolist(),
        "nulls_agree": agree,
        "n_resolved_comparisons": resolved_count,
        "n_unresolved_comparisons": len(rows) - resolved_count,
        "n_checkpoints": len(rows),
        # what the rows are. a reader, and the verifier, must not have to guess
        # whether these are correlated snapshots or independent runs.
        "unit": "independent_agent" if replicate_mode else "checkpoint_of_one_run",
        "agent_seeds": int(args.agent_seeds),
        "steps_per_agent": int(args.steps),
        "actual_steps_per_agent": int(np.ceil(args.steps / 2048) * 2048),
        "null_steps": int(args.null_steps),
        "actual_null_steps": int(np.ceil(args.null_steps / 2048) * 2048),
    }


def run_environments(args):
    """Isolate global training RNG state; preserve the requested task order."""
    args.worker_count = min(args.jobs, len(args.envs))
    if args.worker_count == 1:
        return [run_env(env_id, args) for env_id in args.envs]
    with ProcessPoolExecutor(max_workers=args.worker_count,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_env, env_id, args) for env_id in args.envs]
        return [future.result() for future in futures]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["CartPole-v1", "Acrobot-v1"])
    ap.add_argument("--out", default="reports")
    ap.add_argument("--jobs", type=int, default=1,
                    help="number of isolated worker processes for independent environments")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--null-steps", type=int, default=30_000)
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--attr-episodes", type=int, default=20)
    ap.add_argument("--agent-seeds", type=int, default=1,
                    help="how many INDEPENDENT agents to train per environment. "
                         "1 keeps the reported behaviour, snapshotting a single "
                         "run at four fractions, which gives correlated "
                         "observations of one trajectory. more than 1 trains "
                         "that many agents to full length instead, so the "
                         "replicate is the training run and the spread across "
                         "them is a spread across agents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainer", choices=("native", "sb3"), default="native")
    ap.add_argument("--blind", choices=("gaussian", "resample"), default="gaussian",
                    help="how blinded observations are drawn. gaussian is the "
                         "moment-matched per-coordinate draw the reported "
                         "numbers use; resample draws whole observations and "
                         "so stays on the observation manifold. the default is "
                         "gaussian because changing it silently would make the "
                         "pipeline stop reproducing the paper's table; the "
                         "resample comparison is reported separately")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("jobs must be positive")
    check_resolving_power(args.n_null, what="the generalize gym null")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = run_environments(args)

    banner("ACROSS ENVIRONMENTS")
    print(f"  {'environment':<16}{'weight null sd':>16}{'env null sd':>14}"
          f"{'ratio':>14}{'nulls agree':>12}")
    for r in results:
        ratio = r["width_ratio"]
        # a degenerate null has zero variance, so the ratio is meaningless
        # rather than merely large; say so instead of printing 1e11.
        ratio_s = "undefined" if ratio is None else f"{ratio:.0f}x"
        print(f"  {r['env_id']:<16}{r['weight_null']['std']:>16.2f}"
              f"{r['env_null']['std']:>14.4f}{ratio_s:>14}"
              f"{r['nulls_agree']:>10}/{r['n_resolved_comparisons']}")

    total_pairs = sum(r["n_checkpoints"] for r in results)
    total_agree = sum(r["nulls_agree"] for r in results)
    total_resolved = sum(r["n_resolved_comparisons"] for r in results)
    print(f"\n  resolved reference decisions agree on {total_agree}/{total_resolved} "
          f"comparisons across {len(results)} environments; "
          f"{total_pairs - total_resolved} comparisons unresolved")

    eligible = [row for task in results for row in task["checkpoints"]
                if row["competence"]["eligible_for_sensitivity_claim"]]
    resolved = [row for row in eligible if "unresolved" not in
                (row["verdict_env"], row["verdict_weight"])]
    print(f"  competent observation-dependent agents: {len(eligible)}/{total_pairs}")
    print(f"  eligible agents with two resolved references: {len(resolved)}")
    print("  failed learning and unresolved references do not establish detection sensitivity")

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
        ratio_text = f"{b / a:.2f}" if a > 1e-12 else "undefined"
        print(f"  {r['env_id']:<18}{a:>18.2f}{b:>17.2f}{ratio_text:>9}")

    conj: dict = {"n_environments": len(results)}
    print("  report within-task width ratios; cross-task native rewards have different units")
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
