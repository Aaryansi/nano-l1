"""Single-seed signal-strength sweep and a separate ranking-gap diagnostic.

The filename is retained for existing pipeline calls. Rejection rates over
independent trials are measured by calibrate_detection.py. This sweep neither
estimates statistical power nor identifies a population detection threshold.
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
from nano_rl.env.synthetic import synthetic_train_eval, make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background, masked_span  # noqa: E402
from nano_rl.explain.sanity import (  # noqa: E402
    attribution_span,
    ranking_gap_diagnostic,
    check_resolving_power,
    short_verdict,
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
    reference = json.loads(sj.read_text())
    if reference.get("training_updates") != args.updates:
        raise SystemExit("reference training budget differs; generate a matching reference first")
    null_spans = np.array(reference["null_spans"], dtype=float)
    print(f"null distribution: {len(null_spans)} samples, "
          f"{null_spans.mean():+.3f} +/- {null_spans.std(ddof=1):.3f}")
    # this budget is inherited from whatever wrote sanity_test.json rather than
    # chosen here, so it is checked rather than assumed: a sweep whose every
    # point is unresolved would still plot a smooth curve.
    check_resolving_power(len(null_spans), what="the power-curve null")

    # ------------------------------------------------------------- part A
    banner("A. SIGNAL-STRENGTH SWEEP: one training seed per strength")
    strengths = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
    rows = []

    for st in strengths:
        pair = synthetic_train_eval(make_learnable_corpus,
            n_episodes=1200, signal_strength=st, seed=args.seed)
        batch = pair.train
        norm = fit_normalizer(batch)
        agent = train(batch, norm, args.updates, args.seed)

        ev = BinaryMarketEnv(
            pair.evaluation, max_position=100.0, normalizer=norm, random_episode_order=False
        )
        edge = float(agent.evaluate(ev, n_episodes=len(pair.evaluation))["returns"].mean())

        roll = VectorizedRollout(pair.evaluation, normalizer=norm, max_position=100.0)
        bg = build_background(roll, n_samples=256, seed=args.seed)
        res = test_span_against_null(masked_span(agent, roll, bg, args.seed), null_spans)
        rows.append(
            {
                "signal_strength": st,
                "data": pair.metadata(),
                "agent_edge": edge,
                "span": res.statistic,
                "z": res.z_score,
                "p_rank": res.p_rank,
                "detected": bool(res.passes),
                "verdict": short_verdict(res),
            }
        )
        print(
            f"  strength {st:>4.2f}  edge {edge:>+8.2f}/ep  span {res.statistic:>+8.2f}  "
            f"z {res.z_score:>+7.2f}  {short_verdict(res)}",
            flush=True,
        )

    detected = [r for r in rows if r["detected"]]
    if detected:
        thresh = min(r["agent_edge"] for r in detected)
        print(f"\n  Smallest measured edge with a rejection in this single-seed sweep: {thresh:+.2f}")
        print("  This does not estimate a population detection threshold or power.")
    else:
        thresh = float("nan")
        print("\n  No rejection in this sweep; unresolved outcomes remain distinct.")

    plots.power_curve(
        [r["agent_edge"] for r in rows],
        [r["z"] for r in rows],
        [r["detected"] for r in rows],
        out / "power_curve.png",
        real_edge=None,
        subtitle=f"reference from {len(null_spans)} independently trained null agents; "
                 f"rank and normal criteria both required",
        verdicts=[r["verdict"] for r in rows],
    )

    # ------------------------------------------------------------- part B
    banner("B. RANKING ESTIMATION AND A SEPARATE REFERENCE COMPARISON")
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
    # the adjacent-gap diagnostic uses.
    beh, action = explain_behaviour(
        agent, obs, bg, n_permutations=600, seed=args.seed
    )
    cert = ranking_gap_diagnostic(beh, k=5)

    print("  top-5 ranking and its next-feature boundary, with estimation error:")
    for i, (f, v, e) in enumerate(
        zip(cert["features"], cert["values"], cert["stderr"])
    ):
        print(f"    {i + 1}. {f:<24} {v:>8.4f} +/- {e:.4f}")
    sep = cert["adjacent_pairs_separated"]
    print(f"\n  adjacent pairs separated beyond their combined error: "
          f"{sum(sep)}/{len(sep)}")
    print(f"  all inspected gaps separated (not a formal certificate): {cert['all_gaps_separated']}")

    outc = explain_outcomes(
        agent, split.test, bg, normalizer=split.normalizer,
        cfg=OutcomeAttributionConfig(
            n_coalitions=args.coalitions, n_episodes=args.episodes, seed=args.seed
        ),
    )
    res_real = test_span_against_null(attribution_span(outc), null_spans)
    print(f"\n  and the same agent's explanation against the null:")
    print(f"    {res_real.summary()}")

    print("  The gap diagnostic and reference comparison measure different properties.")
    print("  Neither establishes attribution accuracy or absence of policy information.")

    payload = {
        "null_spans": null_spans.tolist(),
        "training_updates": args.updates,
        "power_curve": rows,
        "detection_threshold_edge": thresh,
        "design": "one training seed per strength; an effect-size sweep, not a power estimate",
        "ranking_gap_diagnostic": cert,
        "real_agent_null_test": res_real.as_dict(),
        "real_agent_null_test_scope": {"checkpoint": ckpts[0].name,
                                       "test_episodes": min(args.episodes, len(split.test)),
                                       "role": "subset diagnostic, not the primary full-test endpoint"},
    }
    (out / "power_curve.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/power_curve.json")


if __name__ == "__main__":
    main()
