"""phase 5: shapley explanations, validated against ground truth.

three sections, in order of what they establish.

  1. GROUND TRUTH. train an agent on the synthetic learnable corpus, where
     exactly one of the eighteen features carries information by construction.
     correct attributions must concentrate on that feature and give the other
     seventeen approximately nothing. this is the check most explainability
     work cannot run, because real data has no known answer, and it is what
     licenses believing anything in sections 2 and 3.

  2. THE NAIVE EXPLANATION IS INCOMPLETE. contrast per-decision attribution of
     pi(a|s) at a mid-episode step against trajectory-aware attribution of the
     episode return. the spec asks for one concrete case where they clearly
     disagree; this produces it with ground truth available to say which is
     right.

  3. THE REAL AGENT. all three targets (behaviour, outcomes, value) applied to
     the ppo agent trained on kalshi data, with figures.

usage:
    python scripts/explain.py --corpus data/corpus/corpus_candles_60s_spot.npz \\
        --runs runs/ppo --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl import plots  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import ACTION_NAMES, BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    SIGNAL_OBS_IDX,
    fit_normalizer,
)
from nano_rl.env.synthetic import make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.trajectory import (  # noqa: E402
    OutcomeAttributionConfig,
    compare_naive_and_trajectory,
    explain_behaviour,
    explain_outcomes,
    explain_value,
)

SIGNAL_NAME = FEATURE_NAMES[SIGNAL_OBS_IDX]


def banner(text: str) -> None:
    print(f"\n{'=' * 76}\n{text}\n{'=' * 76}", flush=True)


# ---------------------------------------------------------------- section 1
def ground_truth_validation(out: Path, updates: int, seed: int) -> dict:
    banner("1. GROUND TRUTH: can the attribution find a feature we planted?")
    print(f"  corpus: synthetic. `{SIGNAL_NAME}` is the ONLY feature that")
    print(f"  carries information about SETTLEMENT. the other {N_FEATURES - 1} are")
    print("  noise or constants by construction.")
    print()
    print("  note the precise claim. other features can still legitimately affect")
    print("  the RETURN without predicting the outcome, because return depends on")
    print("  trading behaviour too: an agent that cannot tell where it is in the")
    print("  episode trades incoherently and pays fees. so the test is that the")
    print("  planted feature ranks FIRST and carries the majority of the mass,")
    print("  not that everything else is exactly zero.\n")

    batch = make_learnable_corpus(n_episodes=1500, seed=seed)
    norm = fit_normalizer(batch)

    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)

    eval_env = BinaryMarketEnv(
        batch, max_position=100.0, normalizer=norm, random_episode_order=False
    )
    perf = agent.evaluate(eval_env, n_episodes=500)
    print(f"  trained agent earns {perf['returns'].mean():.2f} per episode")

    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    background = build_background(roll, n_samples=256, seed=seed)

    # OUTCOMES target: what earned the money?
    att = explain_outcomes(
        agent, batch, background, normalizer=norm,
        cfg=OutcomeAttributionConfig(n_coalitions=200, n_episodes=300, seed=seed),
    )

    mass = np.abs(att.values)
    share = mass[SIGNAL_OBS_IDX] / mass.sum() if mass.sum() > 0 else 0.0

    print(f"\n  attribution of episode return:")
    for name, val in att.top(5):
        marker = "  <-- the planted signal" if name == SIGNAL_NAME else ""
        print(f"    {name:<24} {val:+8.3f}{marker}")

    print(f"\n  share of total |attribution| on the planted feature: {share:.1%}")
    print(f"  v(no features) = {att.base_value:+.3f}, "
          f"v(all features) = {att.full_value:+.3f}")
    print(f"  efficiency gap: {att.efficiency_gap:.2e}")

    verdict = "PASS" if share > 0.5 else "FAIL"
    print(f"\n  ground-truth check: {verdict} "
          f"(need the planted feature to carry the majority of the mass)")

    plots.attribution_bar(
        list(FEATURE_NAMES), att.values,
        out / "attribution_ground_truth.png",
        "validation: attribution of return on a corpus with one planted signal",
        subtitle=f"`{SIGNAL_NAME}` carries {share:.0%} of the attribution mass; "
                 f"the other {N_FEATURES-1} features are noise by construction",
    )
    return {
        "signal_share": float(share),
        "verdict": verdict,
        "agent_return": float(perf["returns"].mean()),
        "top": att.top(5),
    }


# ---------------------------------------------------------------- section 2
def naive_versus_trajectory(out: Path, updates: int, seed: int) -> dict:
    banner("2. WHERE THE PER-DECISION EXPLANATION IS INCOMPLETE")
    print("  the agent's optimal policy here is: read the signal at step 0,")
    print("  take a position, then hold. so at a MID-EPISODE step the action")
    print("  'hold' is driven by the inventory it already has, while the money")
    print("  was earned by the signal several steps earlier.\n")

    batch = make_learnable_corpus(n_episodes=1500, seed=seed)
    norm = fit_normalizer(batch)

    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)

    # search for a state where the agent is genuinely HOLDING an open
    # position mid-episode. an earlier version explained a fixed step 7 of
    # episode 0 and found the agent flat there, which makes the intended
    # contrast vacuous: "why are you holding" is not a question you can ask an
    # agent that holds nothing.
    import torch

    eval_env = BinaryMarketEnv(
        batch, max_position=100.0, normalizer=norm, random_episode_order=False
    )

    found = None
    for ep in range(200):
        obs, _ = eval_env.reset(options={"episode": ep})
        for t in range(eval_env.n_steps):
            with torch.no_grad():
                a, _, _ = agent.net.act(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                    deterministic=True,
                )
            nxt, _, done, _, info = eval_env.step(int(a.item()))
            # a holding state: inventory is open and it was opened earlier
            if abs(info["position"]) > 0 and t >= 4 and not done:
                found = (ep, t + 1, nxt, info["position"])
                break
            obs = nxt
            if done:
                break
        if found:
            break

    if found is None:
        print("  no mid-episode holding state found; cannot run this contrast")
        return {"status": "no holding state found"}

    ep_idx, mid_step, obs, held = found
    print(f"  found: episode {ep_idx}, step {mid_step}, "
          f"holding {held:+.0f} contracts")

    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    background = build_background(roll, n_samples=256, seed=seed)

    naive, action = explain_behaviour(agent, obs, background, n_permutations=250, seed=seed)
    print(f"  the agent's action here is {ACTION_NAMES[action]}\n")

    traj = explain_outcomes(
        agent, batch, background, normalizer=norm,
        cfg=OutcomeAttributionConfig(n_coalitions=200, n_episodes=300, seed=seed),
    )

    cmp = compare_naive_and_trajectory(naive, traj)

    print("  per-decision attribution of pi(a|s) at this step:")
    for name, val in cmp["naive_top"][:4]:
        print(f"    {name:<24} {val:+8.4f}")
    print("\n  trajectory-aware attribution of the episode return:")
    for name, val in cmp["trajectory_top"][:4]:
        print(f"    {name:<24} {val:+8.4f}")

    print(f"\n  rank correlation between the two: {cmp['rank_correlation']:+.3f}")
    print(f"  top-5 overlap: {cmp['top_k_overlap']:.0%}")
    if cmp["only_in_naive"]:
        print(f"  credited ONLY by the per-decision view : {cmp['only_in_naive']}")
    if cmp["only_in_trajectory"]:
        print(f"  credited ONLY by the trajectory view   : {cmp['only_in_trajectory']}")

    naive_rank = int(np.argsort(-np.abs(naive.values)).tolist().index(SIGNAL_OBS_IDX)) + 1
    traj_rank = int(np.argsort(-np.abs(traj.values)).tolist().index(SIGNAL_OBS_IDX)) + 1
    print(f"\n  the planted signal `{SIGNAL_NAME}` ranks:")
    print(f"    {naive_rank:>2} of {N_FEATURES} under per-decision attribution")
    print(f"    {traj_rank:>2} of {N_FEATURES} under trajectory-aware attribution")
    print("\n  ground truth says it should rank 1: it is the ONLY feature that")
    print("  carries information. the per-decision view is not wrong about the")
    print("  action, it is answering a different and narrower question.")

    plots.attribution_comparison(
        list(FEATURE_NAMES), naive.values, traj.values,
        out / "attribution_naive_vs_trajectory.png",
        subtitle=f"episode {ep_idx} step {mid_step}, agent holding "
                 f"{held:+.0f} contracts; rank correlation "
                 f"{cmp['rank_correlation']:+.2f}",
    )
    return {
        "rank_correlation": cmp["rank_correlation"],
        "top_k_overlap": cmp["top_k_overlap"],
        "signal_rank_naive": naive_rank,
        "signal_rank_trajectory": traj_rank,
        "action": ACTION_NAMES[action],
    }


# ---------------------------------------------------------------- section 3
def real_agent(corpus: str, runs: str, out: Path, seed: int) -> dict:
    banner("3. THE REAL AGENT: three explanation targets")

    batch = EpisodeBatch.load(corpus)
    split = walk_forward_split(batch)

    ckpt = sorted(Path(runs).glob("seed*.pt"))
    if not ckpt:
        print(f"  no checkpoints in {runs}; skipping")
        return {}
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpt[0]))
    print(f"  loaded {ckpt[0].name}")

    roll = VectorizedRollout(
        split.test, normalizer=split.normalizer, max_position=100.0
    )
    background = build_background(roll, n_samples=256, seed=seed)

    env = BinaryMarketEnv(
        split.test, normalizer=split.normalizer, max_position=100.0,
        random_episode_order=False,
    )
    obs, _ = env.reset(options={"episode": 0})

    # BEHAVIOUR
    beh, action = explain_behaviour(agent, obs, background, n_permutations=250, seed=seed)
    print(f"\n  BEHAVIOUR: why {ACTION_NAMES[action]} at the first decision?")
    for name, val in beh.top(4):
        print(f"    {name:<24} {val:+8.4f}")
    plots.attribution_bar(
        list(FEATURE_NAMES), beh.values, out / "attribution_behaviour.png",
        f"behaviour: what drives the policy toward {ACTION_NAMES[action]}",
        stderr=beh.stderr,
        subtitle="attribution of pi(a|s) at a single state, test split",
    )

    # VALUE
    val_att = explain_value(agent, obs, background, n_permutations=250, seed=seed)
    print("\n  VALUE: what drives the critic's estimate of this state?")
    for name, v in val_att.top(4):
        print(f"    {name:<24} {v:+8.4f}")
    plots.attribution_bar(
        list(FEATURE_NAMES), val_att.values, out / "attribution_value.png",
        "value: what drives the critic's V(s)",
        stderr=val_att.stderr,
        subtitle="attribution of the critic output at a single state, test split",
    )

    # OUTCOMES
    outc = explain_outcomes(
        agent, split.test, background, normalizer=split.normalizer,
        cfg=OutcomeAttributionConfig(n_coalitions=200, n_episodes=400, seed=seed),
    )
    print("\n  OUTCOMES: what drives the return actually earned?")
    for name, v in outc.top(4):
        print(f"    {name:<24} {v:+8.4f}")
    print(f"    v(blind) = {outc.base_value:+.3f}, "
          f"v(full observation) = {outc.full_value:+.3f}")
    print(f"    so the whole observation is worth "
          f"{outc.full_value - outc.base_value:+.3f} per episode")
    plots.attribution_bar(
        list(FEATURE_NAMES), outc.values, out / "attribution_outcomes.png",
        "outcomes: what drives the episode return",
        subtitle=f"trajectory-aware; observing everything is worth "
                 f"{outc.full_value - outc.base_value:+.3f} per episode "
                 f"against being blind",
    )

    cmp = compare_naive_and_trajectory(beh, outc)
    print(f"\n  behaviour vs outcomes rank correlation: "
          f"{cmp['rank_correlation']:+.3f}")

    return {
        "behaviour_top": beh.top(4),
        "value_top": val_att.top(4),
        "outcomes_top": outc.top(4),
        "value_of_observation": float(outc.full_value - outc.base_value),
        "behaviour_vs_outcomes_rank_corr": cmp["rank_correlation"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "ground_truth": ground_truth_validation(out, args.updates, args.seed),
        "naive_vs_trajectory": naive_versus_trajectory(out, args.updates, args.seed),
        "real_agent": real_agent(args.corpus, args.runs, out, args.seed),
    }
    (out / "explanations.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}/explanations.json")


if __name__ == "__main__":
    main()
