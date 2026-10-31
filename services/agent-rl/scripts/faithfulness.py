"""Descriptive decoy attribution and return-deletion diagnostics.

The decoy generator removes settlement dependence in its held-out population.
That design does not establish what a trained policy learned or supply exact
feature contributions to an input-replacement return game. Policy actions,
trading costs, state dependence, finite backgrounds and coalition interactions
can matter even when a feature does not predict settlement. Probability and
return attributions also explain different targets.

Deletion curves compare return sensitivity under the specified replacement
intervention. Their area ranks these measured curves, not explanation fidelity
in general. The random ranking is one comparison, not an uncertainty estimate.
Neither experiment supplies exact coalition-value ground truth.

usage:
    python scripts/faithfulness.py --out reports
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
from nano_rl.env.binary_market import BinaryMarketEnv  # noqa: E402
from nano_rl.env.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    N_KALSHI,
    fit_normalizer,
)
from nano_rl.env.synthetic import (  # noqa: E402
    DECOY_IDX,
    make_decoy_corpus,
    make_learnable_corpus, synthetic_train_eval,
)
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.trajectory import (  # noqa: E402
    OutcomeAttributionConfig,
    explain_behaviour,
    explain_outcomes,
)

DECOY_OBS_IDX = N_KALSHI + DECOY_IDX
DECOY_NAME = FEATURE_NAMES[DECOY_OBS_IDX]


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def train_on(batch, norm, updates: int, seed: int) -> PPOAgent:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def deletion_curve(
    agent: PPOAgent,
    batch,
    norm,
    background: np.ndarray,
    ranking: np.ndarray,
    max_remove: int = 10,
) -> np.ndarray:
    """mean return after masking the first k features of `ranking`."""
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    rng = np.random.default_rng(0)
    out = []

    for k in range(max_remove + 1):
        removed = ranking[:k]
        mask = np.ones(N_FEATURES, dtype=bool)
        mask[removed] = False

        def policy(obs: np.ndarray) -> np.ndarray:
            synthetic = obs.copy()
            if k > 0:
                draws = background[rng.integers(0, len(background), size=len(obs))]
                synthetic[:, ~mask] = draws[:, ~mask]
            with torch.no_grad():
                logits, _ = agent.net(
                    torch.as_tensor(synthetic, dtype=torch.float32)
                )
                return logits.argmax(dim=-1).numpy()

        out.append(float(roll.run(policy)["returns"].mean()))

    return np.array(out)


# ------------------------------------------------------------------- part A
def decoy_experiment(out: Path, updates: int, seed: int) -> dict:
    banner("A. THE DECOY: compare attribution targets after a distribution shift")

    n_episodes, train_frac = 3000, 0.6
    batch = make_decoy_corpus(
        n_episodes=n_episodes, train_frac=train_frac, seed=seed
    )
    n_train = int(n_episodes * train_frac)

    in_sample = batch.subset(np.arange(n_train))
    held_out = batch.subset(np.arange(n_train, n_episodes))

    d = batch.spot[:, 0, DECOY_IDX]
    print(f"  `{DECOY_NAME}` is carrying the decoy.")
    print(f"    correlation with settlement, in-sample : "
          f"{np.corrcoef(d[:n_train], batch.settlement[:n_train])[0, 1]:+.4f}")
    print(f"    correlation with settlement, held-out  : "
          f"{np.corrcoef(d[n_train:], batch.settlement[n_train:])[0, 1]:+.4f}")
    print("    the generator makes the held-out decoy independent of settlement.")
    print("    sample correlation does not identify its return contribution.\n")

    norm = fit_normalizer(in_sample)
    agent = train_on(in_sample, norm, updates, seed)

    def perf(b):
        env = BinaryMarketEnv(
            b, max_position=100.0, normalizer=norm, random_episode_order=False
        )
        r = agent.evaluate(env, n_episodes=min(600, len(b)))
        return float(r["returns"].mean()), float(r["trades"].mean())

    in_ret, in_tr = perf(in_sample)
    out_ret, out_tr = perf(held_out)
    print(f"  agent return in-sample : {in_ret:+8.2f}  ({in_tr:.2f} trades/ep)")
    print(f"  agent return held-out  : {out_ret:+8.2f}  ({out_tr:.2f} trades/ep)")
    print("  these returns alone do not establish learned feature reliance.\n")

    roll = VectorizedRollout(held_out, normalizer=norm, max_position=100.0)
    background = build_background(roll, n_samples=256, seed=seed)

    # attribute the POLICY over many states, not one. a single-state
    # attribution is dominated by where that particular state happens to sit
    # relative to the background: an earlier version explained step 0 of
    # episode 0 and gave the decoy 0.7% of the mass, which says nothing about
    # whether the policy relies on it in general. the standard remedy is the
    # global summary, i.e. the mean absolute attribution over a sample of
    # states, so that is what is compared against the trajectory value.
    env = BinaryMarketEnv(
        held_out, max_position=100.0, normalizer=norm, random_episode_order=False
    )

    n_states = 40
    acc = np.zeros(N_FEATURES)
    rng_states = np.random.default_rng(seed)
    for i in range(n_states):
        ep = int(rng_states.integers(0, len(held_out)))
        obs, _ = env.reset(options={"episode": ep})
        # step to a random point in the episode so the sample is not all step 0
        for _ in range(int(rng_states.integers(0, env.n_steps - 1))):
            with torch.no_grad():
                a, _, _ = agent.net.act(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                    deterministic=True,
                )
            obs, _, done, _, _ = env.step(int(a.item()))
            if done:
                break
        att_i, _ = explain_behaviour(
            agent, obs, background, n_permutations=60, seed=seed + i
        )
        acc += np.abs(att_i.values)
    acc /= n_states

    from nano_rl.explain.shapley import Attribution

    naive = Attribution(
        values=acc,
        stderr=np.zeros(N_FEATURES),
        base_value=0.0,
        full_value=float(acc.sum()),
        feature_names=FEATURE_NAMES,
    )
    action = 0
    print(f"  per-decision attribution averaged over {n_states} sampled states\n")
    traj = explain_outcomes(
        agent, held_out, background, normalizer=norm,
        cfg=OutcomeAttributionConfig(n_coalitions=220, n_episodes=400, seed=seed),
    )

    def share(att) -> float:
        m = np.abs(att.values)
        return float(m[DECOY_OBS_IDX] / m.sum()) if m.sum() > 0 else 0.0

    def rank(att) -> int:
        return int(np.argsort(-np.abs(att.values)).tolist().index(DECOY_OBS_IDX)) + 1

    print(f"  attribution of the decoy on HELD-OUT episodes:")
    print(f"    {'':<26}{'value':>10}{'share':>9}{'rank':>7}")
    print(f"    {'per-decision, pi(a|s)':<26}"
          f"{naive.values[DECOY_OBS_IDX]:>+10.4f}{share(naive):>8.1%}"
          f"{rank(naive):>7}")
    print(f"    {'trajectory-aware, return':<26}"
          f"{traj.values[DECOY_OBS_IDX]:>+10.4f}{share(traj):>8.1%}"
          f"{rank(traj):>7}")
    print()
    print("  settlement independence does not supply an exact return attribution.")
    print("  these targets and units differ; neither ranking validates the other.")

    # Retain the original ranking pattern as a descriptive observation only.
    rank_pattern = rank(naive) <= 5 and rank(traj) > rank(naive)
    print(f"\n  top-five behaviour rank and lower return rank observed: {rank_pattern}")

    plots.attribution_comparison(
        list(FEATURE_NAMES), naive.values, traj.values,
        out / "attribution_decoy.png",
        title="decoy shares under different attribution targets",
        subtitle=f"held-out decoy/settlement sample correlation "
                 f"{np.corrcoef(d[n_train:], batch.settlement[n_train:])[0,1]:+.3f}; not a fidelity test",
    )

    return {
        "in_sample_return": in_ret,
        "held_out_return": out_ret,
        "decoy_naive_share": share(naive),
        "decoy_trajectory_share": share(traj),
        "decoy_naive_rank": rank(naive),
        "decoy_trajectory_rank": rank(traj),
        "rank_pattern_observed": rank_pattern,
        "scope": "descriptive target comparison; no exact return-attribution ground truth or fidelity verdict",
        "agent": agent,
        "held_out": held_out,
        "norm": norm,
        "background": background,
        "naive": naive,
        "traj": traj,
    }


# ------------------------------------------------------------------- part B
def deletion_experiment(out: Path, updates: int, seed: int) -> dict:
    banner("B. DELETION CURVES: return sensitivity to ranked input replacement")
    print("  features are removed in the order each ranking calls important.")
    print("  lower area measures greater return reduction under this intervention.")
    print("  one random ranking is a comparison, not an uncertainty estimate.\n")

    pair = synthetic_train_eval(make_learnable_corpus, n_episodes=1500, seed=seed)
    batch = pair.train
    norm = fit_normalizer(batch)
    agent = train_on(batch, norm, updates, seed)
    batch = pair.evaluation

    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    background = build_background(roll, n_samples=256, seed=seed)

    env = BinaryMarketEnv(
        batch, max_position=100.0, normalizer=norm, random_episode_order=False
    )
    obs, _ = env.reset(options={"episode": 0})

    naive, _ = explain_behaviour(agent, obs, background, n_permutations=250, seed=seed)
    traj = explain_outcomes(
        agent, batch, background, normalizer=norm,
        cfg=OutcomeAttributionConfig(n_coalitions=220, n_episodes=400, seed=seed),
    )

    rank_naive = np.argsort(-np.abs(naive.values))
    rank_traj = np.argsort(-np.abs(traj.values))
    rank_rand = np.random.default_rng(seed).permutation(N_FEATURES)

    sub = batch.subset(np.arange(400))
    max_remove = 8
    curves = {
        "trajectory-aware": deletion_curve(
            agent, sub, norm, background, rank_traj, max_remove
        ),
        "per-decision": deletion_curve(
            agent, sub, norm, background, rank_naive, max_remove
        ),
        "random (control)": deletion_curve(
            agent, sub, norm, background, rank_rand, max_remove
        ),
    }

    print(f"  {'k removed':>10} " + " ".join(f"{n:>18}" for n in curves))
    for k in range(max_remove + 1):
        print(f"  {k:>10} " + " ".join(f"{c[k]:>18.2f}" for c in curves.values()))

    # Area summarizes this measured intervention curve in return-step units.
    aucs = {n: float(np.trapezoid(c)) for n, c in curves.items()}
    print("\n  area under the measured deletion curve:")
    for n, a in sorted(aucs.items(), key=lambda kv: kv[1]):
        print(f"    {n:<20} {a:>10.1f}")

    better = min(aucs, key=aucs.get)
    print(f"\n  ranking with lowest measured deletion area: {better}")

    plots.deletion_curves(
        curves,
        out / "deletion_curves.png",
        title="return after replacing the top-k ranked features",
        subtitle="specified input replacement; lower area is not a general fidelity guarantee",
    )
    return {"auc": aucs, "lowest_deletion_auc": better,
            "scope": "descriptive return sensitivity; targets differ; one random ranking; no fidelity guarantee",
            "curves": {k: v.tolist() for k, v in curves.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    decoy = decoy_experiment(out, args.updates, args.seed)
    deletion = deletion_experiment(out, args.updates, args.seed)

    payload = {
        "schema_version": 2,
        "decoy": {k: v for k, v in decoy.items()
                  if k not in ("agent", "held_out", "norm", "background",
                               "naive", "traj")},
        "deletion": deletion,
    }
    (out / "faithfulness.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}/faithfulness.json")


if __name__ == "__main__":
    main()
