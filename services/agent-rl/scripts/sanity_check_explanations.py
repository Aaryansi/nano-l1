"""Reference diagnostics for specified input-replacement return spans.

The statistic is full-policy return minus return under a specified replacement
intervention. Shapley efficiency identifies a sum, not feature-level fidelity.
One planted-signal case and one zero-strength case are detection examples, not
estimates of power or specificity. Repeated conditional calibration is separate.

The market checkpoint is evaluated on held-out historical episodes. Its result
against a synthetic reference bank is descriptive: exchangeability with that
bank is not established, and non-rejection does not imply an empty explanation,
absence of exploitable signal, or market efficiency.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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


def endpoint_span(agent, batch, norm, seed, *, raw=None, key=None):
    if raw is not None and key is None:
        raise ValueError("retaining endpoint returns requires a unique key")
    roll = VectorizedRollout(batch, normalizer=norm, max_position=100.)
    background = build_background(roll, n_samples=256, seed=seed)
    result = masked_span(agent, roll, background, seed, return_samples=raw is not None)
    if raw is None:
        return result
    if f"{key}_full" in raw:
        raise ValueError(f"duplicate endpoint key: {key}")
    raw[f"{key}_full"] = result["full"]
    raw[f"{key}_masked"] = result["masked"]
    raw[f"{key}_open_epoch"] = batch.open_epoch
    return result["span"]


def reference_protocol(updates):
    """Procedure shared by the reference bank and zero-strength calibration trials."""
    ppo = asdict(PPOConfig())
    ppo.pop("seed")
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "nano_rl").rglob("*.py")) + [
        Path(__file__).resolve(), root / "scripts/calibrate_detection.py"]
    return {"generator": "make_learnable_corpus(signal_strength=0)",
            "implementation_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sources},
            "runtime": {"numpy": np.__version__, "torch": torch.__version__,
                        "torch_threads": torch.get_num_threads()},
            "episodes_per_corpus": 1200, "steps": 14, "noise": .3, "price": .5, "spread": .02,
            "independent_training_evaluation": True, "training_updates": updates,
            "ppo": ppo, "max_position": 100., "normalizer": "training market features only",
            "evaluation": "greedy, all evaluation episodes", "background_samples": 256,
            "intervention": "fresh whole background row per episode per decision",
            "masking_seed_equals_policy_seed": True}


def build_null_distribution(
    n_null: int, updates: int, coalitions: int, episodes: int, base_seed: int, *, raw=None
) -> list:
    """Reference spans from independently drawn zero-strength populations.

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
        span = endpoint_span(agent, pair.evaluation, nn, reference_seed,
                             raw=raw, key=f"reference_{k}")
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
    print("  agents trained on the specified balanced zero-strength population.")
    print("  training and evaluation corpus draws are independent.")
    print("  market comparisons with this bank remain descriptive.\n")
    raw = {}
    nulls = build_null_distribution(
        args.n_null, args.updates, args.coalitions, args.episodes, args.seed, raw=raw
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
    r_signal = test_span_against_null(endpoint_span(
        la, signal_pair.evaluation, ln, args.seed, raw=raw, key="planted_signal"), nulls)
    print(f"  {r_signal.summary()}")
    results["planted_signal"] = r_signal.as_dict()

    # ----------------------------------------- specificity: must not reject
    banner("2. NULL CORPUS (single negative-control trial)")
    null_pair = synthetic_train_eval(make_null_corpus, n_episodes=1200, seed=9999)
    hb = null_pair.train
    hn = fit_normalizer(hb)
    ha = train(hb, hn, args.updates, args.seed + 77)
    r_null = test_span_against_null(endpoint_span(
        ha, null_pair.evaluation, hn, args.seed, raw=raw, key="held_out_null"), nulls)
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
        spans.append(endpoint_span(a, split.test, split.normalizer, args.seed,
                                   raw=raw, key=f"market_{c.stem}"))
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
            "planted signal": r_signal.as_dict(),
            "null corpus": r_null.as_dict(),
            "real market": r_real.as_dict(),
        },
        null_spans,
        out / "sanity_null_test.png",
        subtitle=f"{args.n_null} agents from the balanced zero-strength population",
    )

    payload = {
        "schema_version": 2,
        "reference_protocol": reference_protocol(args.updates),
        "endpoint_returns_file": "sanity_endpoint_returns.npz",
        "reference_endpoint_keys": [f"reference_{k}" for k in range(args.n_null)],
        "observed_endpoint_keys": {"planted_signal": "planted_signal", "held_out_null": "held_out_null",
                                   "real_market": f"market_{ckpts[0].stem}"},
        "real_endpoint_keys": [f"market_{c.stem}" for c in ckpts],
        "endpoint_masking_seeds": {"planted_signal": args.seed, "held_out_null": args.seed,
                                   "real_market": args.seed},
        "observed_policy_seeds": {"planted_signal": args.seed, "held_out_null": args.seed + 77},
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
                           "reference_masking_seeds": list(range(args.seed + 10000, args.seed + 10000 + args.n_null)),
                           "reference_evaluation_seeds": list(range(1001000, 1001000 + args.n_null))},
    }
    np.savez_compressed(out / payload["endpoint_returns_file"], **raw)
    (out / "sanity_test.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/sanity_test.json")


if __name__ == "__main__":
    main()
