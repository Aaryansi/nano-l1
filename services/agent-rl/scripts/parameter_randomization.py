"""Parameter-dependence checks on market and planted-signal agents.

Cascading and independent layer reinitialization follow the design of Adebayo
et al. (2018), adapted to vectors of absolute behaviour and return attributions.
They test sensitivity to learned parameters. This differs from placing a
trained policy's endpoint span in a bank of random-policy spans.

Neither comparison establishes feature-level fidelity or task utility. An
unprofitable policy can have a faithful and useful explanation. The two agents
and fixed correlation thresholds here are descriptive examples, not a test of
whether a policy is worth explaining. Undefined correlations stay undefined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.networks import orthogonal_init  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.data.splits import walk_forward_split  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch  # noqa: E402
from nano_rl.env.features import N_FEATURES, fit_normalizer  # noqa: E402
from nano_rl.env.synthetic import synthetic_train_eval, make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.trajectory import (  # noqa: E402
    OutcomeAttributionConfig,
    explain_outcomes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steer_explanation import global_attribution  # noqa: E402

# top-down, which is the order adebayo et al. cascade in: closest to the
# output first. the value head is excluded because behaviour-level attribution
# reads the policy head only, so randomizing it would be a no-op that made the
# cascade look deeper than it is.
LAYERS = ["policy_head", "trunk.2", "trunk.0"]


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def _module(net: torch.nn.Module, name: str) -> torch.nn.Module:
    obj = net
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def randomize(net: torch.nn.Module, names: list[str], seed: int) -> None:
    """re-initialise the named layers in place, using the network's own scheme.

    reusing orthogonal_init rather than drawing gaussian noise matters: a
    randomized layer should look like a layer this architecture could have
    started from, otherwise the comparison measures the difference between two
    initialisation schemes as well as the loss of training.
    """
    torch.manual_seed(seed)
    for name in names:
        layer = _module(net, name)
        gain = 0.01 if name == "policy_head" else float(np.sqrt(2))
        orthogonal_init(layer, gain)


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """spearman rank correlation and cosine similarity of two attributions."""
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        rho = float("nan")
    else:
        rho = float(spearmanr(a, b).statistic)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(a @ b / denom) if denom > 0 else float("nan")
    return rho, cos


def similarity_summary(rhos: list[float], cosines: list[float]) -> dict:
    """Summarize defined correlations without treating constants as evidence."""
    def moments(values):
        defined = np.asarray(values, dtype=float)
        defined = defined[np.isfinite(defined)]
        if not len(defined):
            return None, None, 0
        return float(defined.mean()), float(defined.std()), len(defined)

    rho, rho_std, n_rho = moments(rhos)
    cos, cos_std, n_cos = moments(cosines)
    return {"rank_corr_mean": rho, "rank_corr_std": rho_std,
            "cosine_mean": cos, "cosine_std": cos_std,
            "n_defined_rank_corr": n_rho, "n_defined_cosine": n_cos,
            "n_randomizations": len(rhos)}


def descriptive_contrasts(market, planted, market_outcome=None, planted_outcome=None):
    """Fixed exploratory thresholds, not hypothesis tests or target agreement."""
    def defined(*values):
        return all(v is not None and np.isfinite(v) for v in values)

    return {
        "correlation_gap_exceeds_threshold": (
            bool(abs(market - planted) > 0.3) if defined(market, planted) else None),
        "outcome_pattern_threshold_crossed": (
            bool(market_outcome < 0.6 < planted_outcome)
            if defined(market_outcome, planted_outcome) else None),
        "thresholds": {"absolute_correlation_gap": 0.3, "outcome_correlation": 0.6},
        "scope": "descriptive heuristics; undefined correlations cannot establish a finding",
    }


def format_diagnostic(value) -> str:
    return "undefined" if value is None else f"{value:+.3f}"


def outcome_attribution(
    agent: PPOAgent, batch, bg: np.ndarray, norm, coalitions: int,
    episodes: int, seed: int,
) -> np.ndarray:
    """per-feature OUTCOME attribution, the target our null test actually uses.

    the reviewer objection this answers: section 6.2 degrades behaviour-level
    attributions while the null test reads outcome-level ones, so the two could
    be pointing opposite ways because they read different objects rather than
    because they ask different questions. this runs the canonical check on the
    same target the null test uses, at reduced coalition and episode budgets
    because each coalition costs a full batch replay.
    """
    att = explain_outcomes(
        agent, batch, bg, normalizer=norm, max_position=100.0,
        cfg=OutcomeAttributionConfig(
            n_coalitions=coalitions, n_episodes=episodes, seed=seed
        ),
    )
    return np.abs(att.values)


def run_agent_outcomes(
    label: str, agent: PPOAgent, batch, bg: np.ndarray, norm,
    coalitions: int, episodes: int, seeds: int, base_seed: int,
) -> dict:
    """the cascading check only, on the outcome target."""
    banner(f"{label}  [outcome-level target]")
    baseline = outcome_attribution(agent, batch, bg, norm, coalitions,
                                   episodes, base_seed)
    print(f"  baseline attribution mass {baseline.sum():.4f}")

    original = {k: v.detach().clone() for k, v in agent.net.state_dict().items()}
    rows = []
    for depth, layer in enumerate(LAYERS, start=1):
        names = LAYERS[:depth]
        rhos, coss = [], []
        for s_ in range(seeds):
            agent.net.load_state_dict(original)
            randomize(agent.net, names, seed=7_000 + 97 * s_ + depth)
            att = outcome_attribution(agent, batch, bg, norm, coalitions,
                                      episodes, base_seed + s_)
            rho, cos = similarity(baseline, att)
            rhos.append(rho)
            coss.append(cos)
        row = {
            "stage": layer,
            "layers_randomized": names,
            **similarity_summary(rhos, coss),
        }
        rows.append(row)
        print(f"  cascading    {layer:<12} "
              f"rho {format_diagnostic(row['rank_corr_mean'])} "
              f"+/- {format_diagnostic(row['rank_corr_std'])}   "
              f"cos {format_diagnostic(row['cosine_mean'])}")
    agent.net.load_state_dict(original)
    return {"cascading": rows}


def run_agent(
    label: str, agent: PPOAgent, env: BinaryMarketEnv, bg: np.ndarray,
    n_states: int, seeds: int, base_seed: int,
) -> dict:
    banner(f"{label}")

    baseline = global_attribution(agent, env, bg, n_states=n_states, seed=base_seed)
    print(f"  baseline attribution mass {baseline.sum():.4f}")

    original = {k: v.detach().clone() for k, v in agent.net.state_dict().items()}
    out: dict[str, list] = {"cascading": [], "independent": []}

    for mode in ("cascading", "independent"):
        for depth, layer in enumerate(LAYERS, start=1):
            names = LAYERS[:depth] if mode == "cascading" else [layer]
            rhos, coss = [], []
            for s in range(seeds):
                agent.net.load_state_dict(original)
                randomize(agent.net, names, seed=7_000 + 97 * s + depth)
                att = global_attribution(
                    agent, env, bg, n_states=n_states, seed=base_seed + s
                )
                rho, cos = similarity(baseline, att)
                rhos.append(rho)
                coss.append(cos)
            row = {
                "stage": layer,
                "layers_randomized": names,
                **similarity_summary(rhos, coss),
            }
            out[mode].append(row)
            print(f"  {mode:<12} {layer:<12} "
                  f"rho {format_diagnostic(row['rank_corr_mean'])} "
                  f"+/- {format_diagnostic(row['rank_corr_std'])}   "
                  f"cos {format_diagnostic(row['cosine_mean'])}")

    agent.net.load_state_dict(original)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default="runs/ppo")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-states", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outcome-seeds", type=int, default=3)
    ap.add_argument("--outcome-coalitions", type=int, default=64)
    ap.add_argument("--outcome-episodes", type=int, default=150)
    ap.add_argument("--skip-outcomes", action="store_true")
    args = ap.parse_args()

    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    test, norm = split.test, split.normalizer

    ckpt = sorted(Path(args.runs).glob("seed*.pt"))
    if not ckpt:
        raise SystemExit(f"no checkpoints in {args.runs}")
    market = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(ckpt[0]))
    menv = BinaryMarketEnv(test, max_position=100.0, normalizer=norm,
                           random_episode_order=False)
    mroll = VectorizedRollout(test, normalizer=norm, max_position=100.0)
    mbg = build_background(mroll, n_samples=192, seed=args.seed)

    planted_pair = synthetic_train_eval(make_learnable_corpus, n_episodes=1500, seed=args.seed)
    planted = planted_pair.train
    pnorm = fit_normalizer(planted)
    penv_train = BinaryMarketEnv(planted, max_position=100.0, normalizer=pnorm)
    penv_train.reset(seed=args.seed)
    pagent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=args.seed))
    pagent.train(penv_train, n_updates=args.updates, verbose=False)
    planted = planted_pair.evaluation
    penv = BinaryMarketEnv(planted, max_position=100.0, normalizer=pnorm,
                           random_episode_order=False)
    proll = VectorizedRollout(planted, normalizer=pnorm, max_position=100.0)
    pbg = build_background(proll, n_samples=192, seed=args.seed)

    result = {
        "n_states": args.n_states,
        "seeds": args.seeds,
        "checkpoint": ckpt[0].name,
        "updates": args.updates, "seed": args.seed,
        "outcome_seeds": args.outcome_seeds,
        "outcome_coalitions": args.outcome_coalitions,
        "outcome_episodes": args.outcome_episodes,
        "scope": "parameter dependence of two fixed trained agents; not feature fidelity or task utility",
        "market": run_agent("market agent", market, menv,
                            mbg, args.n_states, args.seeds, args.seed),
        "planted": run_agent("planted-signal agent", pagent, penv,
                             pbg, args.n_states, args.seeds, args.seed),
    }

    # the same check on the target the null test actually reads. answers the
    # objection that 6.2 compares behaviour-level degradation against an
    # outcome-level verdict.
    if not args.skip_outcomes:
        result["market_outcomes"] = run_agent_outcomes(
            "market agent", market, test, mbg, norm,
            args.outcome_coalitions, args.outcome_episodes,
            args.outcome_seeds, args.seed)
        result["planted_outcomes"] = run_agent_outcomes(
            "planted-signal agent", pagent, planted, pbg, pnorm,
            args.outcome_coalitions, args.outcome_episodes,
            args.outcome_seeds, args.seed)
        mo = result["market_outcomes"]["cascading"][-1]["rank_corr_mean"]
        po = result["planted_outcomes"]["cascading"][-1]["rank_corr_mean"]
        result["fully_randomized_rank_corr_outcomes"] = {
            "market": mo, "planted": po}

    # the comparison the paper needs: does the check separate the two agents?
    def deepest(d):
        return d["cascading"][-1]["rank_corr_mean"]

    m, p = deepest(result["market"]), deepest(result["planted"])
    outcomes = result.get("fully_randomized_rank_corr_outcomes", {})
    result["descriptive_contrasts"] = descriptive_contrasts(
        m, p, outcomes.get("market"), outcomes.get("planted"))
    result["fully_randomized_rank_corr"] = {"market": m, "planted": p}

    banner("DESCRIPTIVE PARAMETER DEPENDENCE")
    print(f"  fully randomized, market  rho = {format_diagnostic(m)}")
    print(f"  fully randomized, planted rho = {format_diagnostic(p)}")
    print(f"  fixed correlation-gap heuristic: "
          f"{result['descriptive_contrasts']['correlation_gap_exceeds_threshold']}")
    if not args.skip_outcomes:
        o = result["fully_randomized_rank_corr_outcomes"]
        print(f"\n  outcome target, market  rho = {format_diagnostic(o['market'])}")
        print(f"  outcome target, planted rho = {format_diagnostic(o['planted'])}")
    print("\n  the canonical check asks whether an explanation depends on the")
    print("  learned weights. This does not establish feature-level fidelity,")
    print("  task utility, or whether an unprofitable policy is explainable.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parameter_randomization.json").write_text(
        json.dumps(result, indent=2, allow_nan=False))
    print(f"\nwrote {out / 'parameter_randomization.json'}")


if __name__ == "__main__":
    main()
