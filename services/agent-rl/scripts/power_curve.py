"""how much edge must an agent have before its explanation is trustworthy?

the null test in scripts/sanity_check_explanations.py is binary: it rejected a
planted signal at z = +12.27 and failed to reject the real agent at z = -0.15.
that demonstrates the test works, but it does not tell a practitioner anything
actionable. this script calibrates it.

PART A, the power curve. sweep the strength of a planted signal from zero
upward, train an agent at each level, and ask at what point the null test
starts detecting its explanation. the x axis is the agent's MEASURED edge in
dollars per episode, not the latent signal strength, because measured edge is
what a practitioner actually has.

PART B, estimation certainty is not validity. the RankSHAP line of work
certifies that a top-k shapley ranking is stable given monte-carlo noise. that
is a real and useful guarantee about the ESTIMATOR. it says nothing about
whether the explained model learned anything. this part produces a top-k
ranking of the real agent whose adjacent gaps exceed their own standard
errors, and shows that same explanation failing the null test. a precisely
estimated description of nothing.

usage:
    python scripts/power_curve.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.sanity import (  # noqa: E402
    attribution_span,
    certified_top_k,
    test_span_against_null,
)
from nano_rl.explain.trajectory import (  # noqa: E402
    OutcomeAttributionConfig,
    explain_behaviour,
    explain_outcomes,
)


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def train(batch, norm, updates: int, seed: int) -> PPOAgent:
    env = BinaryMarketEnv(batch, max_position=100.0, normalizer=norm)
    env.reset(seed=seed)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
    agent.train(env, n_updates=updates, verbose=False)
    return agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--sanity-json", default=None,
                    help="reuse the null distribution from a previous run")
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--coalitions", type=int, default=140)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # reuse the null rather than recomputing 24 agents. the sweep settings must
    # match the ones the null was built with, or the comparison is invalid.
    sj = Path(args.sanity_json or (out / "sanity_test.json"))
    if not sj.exists():
        print(f"need {sj}; run scripts/sanity_check_explanations.py first")
        raise SystemExit(1)
    null_spans = np.array(json.loads(sj.read_text())["null_spans"], dtype=float)
    print(f"null distribution: {len(null_spans)} samples, "
          f"{null_spans.mean():+.3f} +/- {null_spans.std(ddof=1):.3f}")

    # ------------------------------------------------------------- part A
    banner("A. POWER CURVE: at what edge does the test start detecting?")
    strengths = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
    rows = []

    for st in strengths:
        batch = make_learnable_corpus(
            n_episodes=1200, signal_strength=st, seed=args.seed
        )
        norm = fit_normalizer(batch)
        agent = train(batch, norm, args.updates, args.seed)

        ev = BinaryMarketEnv(
            batch, max_position=100.0, normalizer=norm, random_episode_order=False
        )
        edge = float(agent.evaluate(ev, n_episodes=500)["returns"].mean())

        roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
        bg = build_background(roll, n_samples=256, seed=args.seed)
        att = explain_outcomes(
            agent, batch, bg, normalizer=norm,
            cfg=OutcomeAttributionConfig(
                n_coalitions=args.coalitions, n_episodes=args.episodes,
                seed=args.seed,
            ),
        )
        res = test_span_against_null(attribution_span(att), null_spans)
        rows.append(
            {
                "signal_strength": st,
                "agent_edge": edge,
                "span": res.statistic,
                "z": res.z_score,
                "p_rank": res.p_rank,
                "detected": bool(res.passes),
            }
        )
        print(
            f"  strength {st:>4.2f}  edge {edge:>+8.2f}/ep  span {res.statistic:>+8.2f}  "
            f"z {res.z_score:>+7.2f}  {'DETECTED' if res.passes else 'not detected'}",
            flush=True,
        )

    detected = [r for r in rows if r["detected"]]
    if detected:
        thresh = min(r["agent_edge"] for r in detected)
        print(f"\n  detection threshold: the test flags explanations as informative")
        print(f"  once the agent's measured edge reaches about "
              f"{thresh:+.2f} per episode.")
        print(f"  for reference, the real agent earns -0.418 per episode.")
    else:
        thresh = float("nan")
        print("\n  nothing detected across the sweep")

    plots.power_curve(
        [r["agent_edge"] for r in rows],
        [r["z"] for r in rows],
        [r["detected"] for r in rows],
        out / "power_curve.png",
        real_edge=-0.418,
        subtitle=f"null from {len(null_spans)} agents with nothing to learn; "
                 f"detection at |z| > 1.96",
    )

    # ------------------------------------------------------------- part B
    banner("B. A PRECISELY ESTIMATED DESCRIPTION OF NOTHING")
    print("  the RankSHAP line of work certifies that a top-k shapley ranking")
    print("  is stable given monte-carlo noise. that is a guarantee about the")
    print("  ESTIMATOR. here is what it does not cover.\n")

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpts[0]))

    roll = VectorizedRollout(
        split.test, normalizer=split.normalizer, max_position=100.0
    )
    bg = build_background(roll, n_samples=256, seed=args.seed)

    env = BinaryMarketEnv(
        split.test, normalizer=split.normalizer, max_position=100.0,
        random_episode_order=False,
    )
    obs, _ = env.reset(options={"episode": 0})

    # permutation shapley reports a standard error per feature, which is what
    # the certification needs.
    beh, action = explain_behaviour(
        agent, obs, bg, n_permutations=600, seed=args.seed
    )
    cert = certified_top_k(beh, k=5)

    print("  top-5 ranking of the real agent's policy, with estimation error:")
    for i, (f, v, e) in enumerate(
        zip(cert["features"], cert["values"], cert["stderr"])
    ):
        print(f"    {i + 1}. {f:<24} {v:>8.4f} +/- {e:.4f}")
    sep = cert["adjacent_pairs_separated"]
    print(f"\n  adjacent pairs separated beyond their combined error: "
          f"{sum(sep)}/{len(sep)}")
    print(f"  ranking fully certified: {cert['fully_certified']}")

    outc = explain_outcomes(
        agent, split.test, bg, normalizer=split.normalizer,
        cfg=OutcomeAttributionConfig(
            n_coalitions=args.coalitions, n_episodes=args.episodes, seed=args.seed
        ),
    )
    res_real = test_span_against_null(attribution_span(outc), null_spans)
    print(f"\n  and the same agent's explanation against the null:")
    print(f"    {res_real.summary()}")

    if cert["fully_certified"] and not res_real.passes:
        print("\n  so the ranking is certified stable and the explanation is")
        print("  not distinguishable from an explanation of nothing. the two")
        print("  guarantees are orthogonal, and only one of them is commonly")
        print("  reported.")

    payload = {
        "null_spans": null_spans.tolist(),
        "power_curve": rows,
        "detection_threshold_edge": thresh,
        "certification": cert,
        "real_agent_null_test": res_real.as_dict(),
    }
    (out / "power_curve.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/power_curve.json")


if __name__ == "__main__":
    main()
