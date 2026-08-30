"""why does the weight-randomization null fail in RL? a mechanism, and a test.

section 3.9 measured that the established null has 42x the variance of an
environment-level one on CartPole and is degenerate on Acrobot, and explained it
only as "a randomly initialised policy behaves arbitrarily". that is an
observation, not a mechanism.

the proposed mechanism is structural, and it is the difference between
supervised learning and RL:

    in supervised learning, randomising weights perturbs the MEASUREMENT.
    in RL, it perturbs the measurement's DOMAIN.

return is a functional of the state distribution the policy itself induces. a
random policy visits different states, which changes the return, and that
effect compounds along the trajectory. a random classifier's logit compounds
over nothing. that is why the problem has no supervised analogue: there is no
horizon there.

the prediction is testable. **the weight null's variance should grow with the
episode horizon faster than the environment null's**, so a test built on it
should lose power as horizons lengthen.

why this is measured on the synthetic corpus rather than CartPole. a random
policy on CartPole drops the pole in about twenty steps whatever the time limit
is, so termination, not the horizon, sets the episode length and the horizon
cannot be varied cleanly. the synthetic corpora have fixed-length episodes with
no early termination, so `n_steps` controls the horizon exactly. that confound
is itself worth stating and is reported rather than worked around.

usage:
    python scripts/horizon_scaling.py --out reports
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
from nano_rl.agents.networks import ActorCritic  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import make_null_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def span_of(net, batch, norm, background, rng) -> float:
    """v(all) - v(none) for a raw network, in two masked rollouts."""
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)

    def run(mask: np.ndarray) -> float:
        def policy(obs: np.ndarray) -> np.ndarray:
            x = obs.copy()
            if not mask.all():
                draws = background[rng.integers(0, len(background), size=len(obs))]
                x[:, ~mask] = draws[:, ~mask]
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(x, dtype=torch.float32))
                return logits.argmax(dim=-1).numpy()

        return float(roll.run(policy)["returns"].mean())

    return run(np.ones(N_FEATURES, dtype=bool)) - run(np.zeros(N_FEATURES, dtype=bool))


def measure_horizon(n_steps: int, n_null: int, updates: int, n_episodes: int,
                    seed: int) -> dict:
    """both null distributions at one horizon, everything else held fixed."""
    batch = make_null_corpus(n_episodes=n_episodes, n_steps=n_steps, seed=seed)
    norm = fit_normalizer(batch)
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    bg = build_background(roll, n_samples=192, seed=seed)

    # null 1: randomised weights, the established construction
    weight = []
    for k in range(n_null):
        torch.manual_seed(10_000 + k)
        net = ActorCritic(N_FEATURES, 3, 64)
        weight.append(span_of(net, batch, norm, bg, np.random.default_rng(seed + k)))

    # null 2: trained normally where there is nothing to learn
    env_null = []
    for k in range(n_null):
        nb = make_null_corpus(n_episodes=n_episodes, n_steps=n_steps, seed=5000 + k)
        nn = fit_normalizer(nb)
        e = BinaryMarketEnv(nb, max_position=100.0, normalizer=nn)
        e.reset(seed=seed + k)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed + k))
        agent.train(e, n_updates=updates, verbose=False)
        env_null.append(
            span_of(agent.net, batch, norm, bg, np.random.default_rng(seed + k))
        )

    w, v = np.array(weight), np.array(env_null)
    return {
        "n_steps": n_steps,
        "weight_std": float(w.std(ddof=1)),
        "env_std": float(v.std(ddof=1)),
        "weight_spans": w.tolist(),
        "env_spans": v.tolist(),
        "ratio": float(w.std(ddof=1) / max(v.std(ddof=1), 1e-9)),
    }


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """least-squares exponent of y ~ x^alpha, plus the r^2 of the log fit."""
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    alpha, c = np.polyfit(lx, ly, 1)
    pred = alpha * lx + c
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return float(alpha), 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 7, 14, 28, 56])
    ap.add_argument("--n-null", type=int, default=16)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    banner("MECHANISM: does the weight null's variance grow with the horizon?")
    print("  hypothesis: in RL, randomising weights perturbs the measurement's")
    print("  DOMAIN, because return is a functional of the state distribution")
    print("  the policy induces. that compounds along the trajectory. a random")
    print("  classifier's logit compounds over nothing, which is why this has")
    print("  no supervised analogue.")
    print()
    print("  prediction: weight-null variance grows with horizon faster than")
    print("  environment-null variance, so a test built on it loses power as")
    print("  episodes lengthen.\n")

    print(f"  {'horizon':>8}{'weight sd':>13}{'env sd':>11}{'ratio':>10}")
    rows = []
    for h in args.horizons:
        r = measure_horizon(h, args.n_null, args.updates, args.episodes, args.seed)
        rows.append(r)
        print(f"  {h:>8}{r['weight_std']:>13.3f}{r['env_std']:>11.3f}"
              f"{r['ratio']:>10.1f}x", flush=True)

    x = np.array([r["n_steps"] for r in rows], dtype=float)
    w = np.array([r["weight_std"] for r in rows])
    v = np.array([r["env_std"] for r in rows])

    a_w, r2_w = fit_power_law(x, w)
    a_v, r2_v = fit_power_law(x, v)

    banner("SCALING")
    print(f"  weight null      sd ~ horizon^{a_w:+.2f}   (log-log r^2 = {r2_w:.3f})")
    print(f"  environment null sd ~ horizon^{a_v:+.2f}   (log-log r^2 = {r2_v:.3f})")
    print()

    # a power-law exponent is meaningless if the data is not a power law, so
    # the fit quality gates the verdict. an earlier version omitted this and
    # reported "confirmed" off an environment-null fit with r^2 = 0.258.
    MIN_R2 = 0.8
    fits_ok = (r2_w >= MIN_R2) and (r2_v >= MIN_R2)

    if not fits_ok:
        print(f"  at least one fit is not a power law (r^2 {r2_w:.3f} weight, "
              f"{r2_v:.3f} environment).")
        print("  the exponents cannot be compared, so the prediction is not")
        print("  testable on this data as collected. refitting after excluding")
        print("  any collapsed points:")
        keep = v > 1e-3
        if keep.sum() >= 3:
            a_w2, r2_w2 = fit_power_law(x[keep], w[keep])
            a_v2, r2_v2 = fit_power_law(x[keep], v[keep])
            print(f"    weight      alpha {a_w2:+.2f}  r^2 {r2_w2:.3f}")
            print(f"    environment alpha {a_v2:+.2f}  r^2 {r2_v2:.3f}")
            if a_w2 > a_v2 + 0.3:
                verdict = "confirmed on the non-collapsed range"
            else:
                verdict = "NOT supported"
                print()
                print("  the two grow at indistinguishable rates once collapsed")
                print("  points are excluded. the horizon mechanism does not")
                print("  explain the gap measured on CartPole, and the paper")
                print("  must not claim it does.")
        else:
            verdict = "untestable"
    elif a_w > a_v + 0.3:
        print("  the weight null's variance grows FASTER with the horizon,")
        print("  which is the predicted signature.")
        verdict = "confirmed"
    else:
        print("  the two grow at indistinguishable rates. the horizon mechanism")
        print("  is NOT supported.")
        verdict = "NOT supported"

    # separately: did any environment null collapse?
    collapsed = [int(r["n_steps"]) for r in rows if r["env_std"] < 1e-2]
    if collapsed:
        print()
        print(f"  NOTE: the environment null collapsed to zero variance at "
              f"horizon {collapsed}.")
        print("  every blind agent converged to the same policy, so masking")
        print("  changes nothing. that is the same degenerate null seen on")
        print("  Acrobot, and it is a property of the task affording the agent")
        print("  enough time to learn that doing nothing is best.")

    print(f"\n  verdict: {verdict}")

    plots.horizon_scaling(
        x, w, v, out / "horizon_scaling.png",
        alpha_weight=a_w, alpha_env=a_v,
        subtitle=f"{args.n_null} agents per null per horizon; fixed-length "
                 f"episodes so the horizon is controlled exactly",
    )

    (out / "horizon_scaling.json").write_text(json.dumps({
        "rows": rows,
        "alpha_weight": a_w, "r2_weight": r2_w,
        "alpha_env": a_v, "r2_env": r2_v,
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {out}/horizon_scaling.json")


if __name__ == "__main__":
    main()
