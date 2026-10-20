"""does this explanation say anything, or is it an explanation of nothing?

the observation that motivates the test: five independently trained agents
produce highly consistent explanations (rank correlation 0.865, unanimous on
the most important feature) of a policy that earns -0.418 per episode and is
statistically indistinguishable from doing nothing. consistency is widely used
as a proxy for trustworthiness, and here it is high exactly where the
explanation is empty.

this script runs the null-model test on three cases where the right answer is
known or strongly implied, in increasing order of how much it matters:

  PLANTED SIGNAL   one feature is informative by construction. the test must
                   PASS, or it has no power.
  NULL CORPUS      nothing is informative by construction. the test must FAIL
                   to reject, or it has no specificity.
  REAL MARKET      an efficient market, agent with no measurable edge. this is
                   the case the test exists for.

the statistic is the shapley span v(N) - v(empty), the total value of observing
the state, which the efficiency axiom supplies for free.

usage:
    python scripts/sanity_check_explanations.py \\
        --corpus data/corpus/corpus_candles_60s_spot.npz --runs runs/ppo
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
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus, synthetic_train_eval  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background, masked_span  # noqa: E402
from nano_rl.explain.sanity import (  # noqa: E402
    NOT_DISTINGUISHABLE,
    UNRESOLVED,
    check_resolving_power,
    consistency_across_runs,
    short_verdict,
    test_span_against_null,
)
from nano_rl.explain.trajectory import (  # noqa: E402
    OutcomeAttributionConfig,
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


def attribute(agent, batch, norm, seed: int, coalitions: int, episodes: int):
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.0)
    bg = build_background(roll, n_samples=256, seed=seed)
    return explain_outcomes(
        agent, batch, bg, normalizer=norm,
        cfg=OutcomeAttributionConfig(
            n_coalitions=coalitions, n_episodes=episodes, seed=seed
        ),
    )


def endpoint_span(agent, batch, norm, seed):
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.)
    background = build_background(roll, n_samples=256, seed=seed)
    return masked_span(agent, roll, background, seed)


def build_null_distribution(
    n_null: int, updates: int, coalitions: int, episodes: int, base_seed: int
) -> list:
    """attributions for matched agents trained where there is nothing to learn.

    each null agent gets its own corpus seed as well as its own training seed,
    so the distribution reflects variation in the data-generating process too,
    not just in initialisation.
    """
    out = []
    for k in range(n_null):
        pair = synthetic_train_eval(make_null_corpus, n_episodes=1200, seed=1000 + k)
        nb = pair.train
        nn = fit_normalizer(nb)
        reference_seed = base_seed + 10_000 + k
        agent = train(nb, nn, updates, reference_seed)
        span = endpoint_span(agent, pair.evaluation, nn, reference_seed)
        out.append(span)
        print(f"    null {k + 1}/{n_null}: span {span:+8.4f}",
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-null", type=int, default=24)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--coalitions", type=int, default=140)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    check_resolving_power(args.n_null, what="the sanity check explanations null")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    banner("0. THE NULL DISTRIBUTION")
    print("  agents trained where there is provably nothing to learn.")
    print("  this is the reinforcement-learning analogue of a randomization")
    print("  test, but the null is one that actually occurs in deployment:")
    print("  a normally trained agent in a structureless environment.\n")
    nulls = build_null_distribution(
        args.n_null, args.updates, args.coalitions, args.episodes, args.seed
    )
    null_spans = np.asarray(nulls)
    print(f"\n  null span: {null_spans.mean():+.4f} +/- {null_spans.std(ddof=1):.4f}")

    results = {}

    # ------------------------------------------------------ power: must pass
    banner("1. PLANTED SIGNAL (single positive-control trial)")
    signal_pair = synthetic_train_eval(make_learnable_corpus, n_episodes=1200, seed=args.seed)
    lb = signal_pair.train
    ln = fit_normalizer(lb)
    la = train(lb, ln, args.updates, args.seed)
    r_signal = test_span_against_null(endpoint_span(la, signal_pair.evaluation, ln, args.seed), nulls)
    print(f"  {r_signal.summary()}")
    results["planted_signal"] = r_signal.as_dict()

    # ----------------------------------------- specificity: must not reject
    banner("2. NULL CORPUS (single negative-control trial)")
    null_pair = synthetic_train_eval(make_null_corpus, n_episodes=1200, seed=9999)
    hb = null_pair.train
    hn = fit_normalizer(hb)
    ha = train(hb, hn, args.updates, args.seed + 77)
    r_null = test_span_against_null(endpoint_span(ha, null_pair.evaluation, hn, args.seed), nulls)
    print(f"  {r_null.summary()}")
    results["held_out_null"] = r_null.as_dict()

    # --------------------------------------------------- the case of interest
    banner("3. THE REAL AGENT ON THE REAL MARKET")
    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    ckpts = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpts:
        print(f"  no checkpoints in {args.runs}")
        raise SystemExit(1)

    real_atts, spans = [], []
    for c in ckpts:
        a = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(c))
        real_atts.append(
            attribute(a, split.test, split.normalizer, args.seed,
                      args.coalitions, args.episodes)
        )
        spans.append(endpoint_span(a, split.test, split.normalizer, args.seed))
        print(f"  {c.name}: full-test span {spans[-1]:+8.4f}", flush=True)

    per_seed = [test_span_against_null(span, nulls).as_dict() for span in spans]
    r_real = test_span_against_null(spans[0], nulls)  # fixed first seed, never selected by its result
    print(f"\n  {r_real.summary()}")
    results["real_market"] = r_real.as_dict()

    consistency = consistency_across_runs([a.values for a in real_atts])

    banner("VERDICT")
    print(f"  {'case':<22}{'span':>10}{'p':>9}  outcome")
    for label, r in (
        ("planted signal", r_signal),
        ("null corpus", r_null),
        ("real market", r_real),
    ):
        print(f"  {label:<22}{r.statistic:>+10.4f}{r.p_rank:>9.4f}  "
              f"{short_verdict(r)}")

    ok_power = None if r_signal.verdict == UNRESOLVED else r_signal.passes
    ok_spec = None if r_null.verdict == UNRESOLVED else r_null.verdict == NOT_DISTINGUISHABLE
    print(f"\n  single positive-control trial: {short_verdict(r_signal)}")
    print(f"  single negative-control trial: {short_verdict(r_null)}")
    print("  these trials do not estimate power or specificity")
    print(f"  subset attribution rank correlation across seeds: {consistency:+.3f}")

    plots.null_test(
        {
            "planted signal": (r_signal.statistic, r_signal.p_rank),
            "null corpus": (r_null.statistic, r_null.p_rank),
            "real market": (r_real.statistic, r_real.p_rank),
        },
        null_spans,
        out / "sanity_null_test.png",
        subtitle=f"null from {args.n_null} agents trained where there is "
                 f"nothing to learn",
    )

    payload = {
        "null_spans": null_spans.tolist(),
        "training_updates": args.updates,
        "synthetic_episodes": 1200,
        "results": results,
        "real_consistency_rank_corr": consistency,
        "real_spans_per_seed": spans,
        "positive_control_detected": ok_power,
        "null_control_not_flagged": ok_spec,
        "real_results_per_seed": per_seed,
        "primary_real_checkpoint": ckpts[0].name,
        "endpoint_evaluation_episodes": len(split.test),
        "feature_attribution_episodes": min(args.episodes, len(split.test)),
        "synthetic_data": {"signal": signal_pair.metadata(), "null": null_pair.metadata(),
                           "reference_train_seeds": list(range(1000, 1000 + args.n_null)),
                           "reference_policy_seeds": list(range(args.seed + 10000, args.seed + 10000 + args.n_null)),
                           "reference_evaluation_seeds": list(range(1001000, 1001000 + args.n_null))},
    }
    (out / "sanity_test.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/sanity_test.json")


if __name__ == "__main__":
    main()
