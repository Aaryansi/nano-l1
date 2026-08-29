"""is the finding a property of RL explanation, or of Shapley values?

every attribution result so far uses Shapley. the obvious objection is that the
failures shown are artefacts of Shapley estimation rather than facts about
explaining RL agents. integrated gradients is the control: it shares no
machinery with Shapley, integrating the model's gradient along a path from a
baseline rather than averaging marginal contributions over coalitions.

both supply a scalar span for free, Shapley by efficiency and IG by
completeness, so the same null test applies to either without modification.

this script also closes the last single-seed gap in the writeup. per-feature
attributions were previously reported from one checkpoint; here both families
are computed across all five trained seeds and reported as mean +/- std.

usage:
    python scripts/second_method.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.features import FEATURE_NAMES, N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus  # noqa: E402
from nano_rl.explain.gradients import ig_attribution_profile, ig_span  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.sanity import (  # noqa: E402
    consistency_across_runs,
    test_span_against_null,
)
from nano_rl.explain.trajectory import explain_behaviour  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def train(batch, norm, updates: int, seed: int) -> PPOAgent:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def sample_states(env: BinaryMarketEnv, agent: PPOAgent, n: int, seed: int) -> np.ndarray:
    """states visited by the agent, spread across episodes and timesteps."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
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
        rows.append(np.asarray(obs, dtype=np.float32))
    return np.asarray(rows[:n])


def shapley_profile(agent, env, background, n_states, seed) -> np.ndarray:
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
        att, _ = explain_behaviour(agent, obs, background, n_permutations=50,
                                   seed=seed + i)
        acc += np.abs(att.values)
    return acc / n_states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=24)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    baseline = np.zeros(N_FEATURES, dtype=np.float32)

    # ------------------------------------------------------ IG null distribution
    banner("A. INTEGRATED GRADIENTS: the same test, a different attribution family")
    print("  IG shares no machinery with shapley. it integrates the model's")
    print("  gradient along a path from a baseline; shapley averages marginal")
    print("  contributions over coalitions. both give a span for free, so the")
    print("  same null test applies unchanged.\n")

    null_spans = []
    for k in range(args.n_null):
        nb = make_null_corpus(n_episodes=1000, seed=1000 + k)
        nn = fit_normalizer(nb)
        agent = train(nb, nn, args.updates, args.seed + k)
        nenv = BinaryMarketEnv(nb, max_position=100.0, normalizer=nn,
                               random_episode_order=False)
        states = sample_states(nenv, agent, args.n_states, args.seed + k)
        null_spans.append(ig_span(agent.net, states, baseline))
    null_spans = np.array(null_spans)
    print(f"  IG null span: {null_spans.mean():+.5f} +/- {null_spans.std(ddof=1):.5f}")

    # power check: a corpus with a planted signal must be detected
    lb = make_learnable_corpus(n_episodes=1000, seed=args.seed)
    ln = fit_normalizer(lb)
    la = train(lb, ln, args.updates, args.seed)
    lenv = BinaryMarketEnv(lb, max_position=100.0, normalizer=ln,
                           random_episode_order=False)
    l_states = sample_states(lenv, la, args.n_states, args.seed)
    r_signal = test_span_against_null(ig_span(la.net, l_states, baseline), null_spans)
    print(f"  planted signal : {r_signal.summary()}")

    # the real agents
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    env = BinaryMarketEnv(split.test, normalizer=split.normalizer,
                          max_position=100.0, random_episode_order=False)
    real_agents = [
        PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(c)) for c in ckpts
    ]

    real_spans, ig_profiles = [], []
    for i, a in enumerate(real_agents):
        st = sample_states(env, a, args.n_states, args.seed + i)
        real_spans.append(ig_span(a.net, st, baseline))
        ig_profiles.append(ig_attribution_profile(a.net, st, baseline, n_steps=64))

    r_real = test_span_against_null(float(np.mean(real_spans)), null_spans)
    print(f"  real agent     : {r_real.summary()}")

    print(f"\n  {'family':<22}{'planted signal':>18}{'real market':>18}")
    print(f"  {'shapley (section 3.5)':<22}{'z = +12.27':>18}{'z = -0.15':>18}")
    print(f"  {'integrated gradients':<22}"
          f"{'z = ' + format(r_signal.z_score, '+.2f'):>18}"
          f"{'z = ' + format(r_real.z_score, '+.2f'):>18}")

    same = (r_signal.passes == True) and (r_real.passes == False)
    print(f"\n  the two families {'AGREE' if same else 'DISAGREE'} on both cases")
    if same:
        print("  so the finding is a property of the agent, not of shapley.")

    # ---------------------------------------------- multi-seed per-feature
    banner("B. PER-FEATURE ATTRIBUTIONS ACROSS ALL FIVE SEEDS")
    print("  closes the last single-seed gap: these were previously reported")
    print("  from one checkpoint.\n")

    roll = VectorizedRollout(split.test, normalizer=split.normalizer,
                             max_position=100.0)
    bg = build_background(roll, n_samples=192, seed=args.seed)

    shap_profiles = [
        shapley_profile(a, env, bg, args.n_states // 2, args.seed + i)
        for i, a in enumerate(real_agents)
    ]

    shap_arr = np.array(shap_profiles)
    ig_arr = np.array(ig_profiles)

    def norm(m):
        t = m.sum(axis=1, keepdims=True)
        return m / np.where(t == 0, 1, t)

    sn, ig_n = norm(shap_arr), norm(ig_arr)

    print(f"  {'feature':<24}{'shapley share':>22}{'IG share':>22}")
    order = np.argsort(-sn.mean(axis=0))[:8]
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<24}"
              f"{sn[:, i].mean():>15.1%} +/-{sn[:, i].std():<5.1%}"
              f"{ig_n[:, i].mean():>15.1%} +/-{ig_n[:, i].std():<5.1%}")

    c_shap = consistency_across_runs(list(shap_arr))
    c_ig = consistency_across_runs(list(ig_arr))
    print(f"\n  cross-seed rank correlation: shapley {c_shap:+.3f}, IG {c_ig:+.3f}")

    # do the two families agree with each other on the ranking?
    agree = float(np.corrcoef(
        np.argsort(np.argsort(-sn.mean(0))), np.argsort(np.argsort(-ig_n.mean(0)))
    )[0, 1])
    print(f"  shapley vs IG rank correlation: {agree:+.3f}")

    plots.method_comparison(
        list(FEATURE_NAMES), sn, ig_n, out / "method_comparison.png",
        subtitle=f"{len(real_agents)} seeds; cross-seed rank correlation "
                 f"{c_shap:+.2f} (shapley) and {c_ig:+.2f} (IG)",
    )

    (out / "second_method.json").write_text(json.dumps({
        "ig_null_spans": null_spans.tolist(),
        "ig_planted_signal": r_signal.as_dict(),
        "ig_real_market": r_real.as_dict(),
        "families_agree": bool(same),
        "shapley_profiles": shap_arr.tolist(),
        "ig_profiles": ig_arr.tolist(),
        "consistency_shapley": c_shap,
        "consistency_ig": c_ig,
        "shapley_vs_ig_rank_corr": agree,
    }, indent=2))
    print(f"\nwrote {out}/second_method.json")


if __name__ == "__main__":
    main()
