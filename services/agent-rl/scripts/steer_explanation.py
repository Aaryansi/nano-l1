"""can an agent's explanation be changed while its performance is held fixed?

this is the safety-relevant question in the project. if an explanation can be
steered without cost, then it reflects a training choice rather than the task,
and an overseer inspecting attributions is inspecting something the developer
controls. the threat model is concrete: a developer who wanted their agent to
appear not to use a sensitive feature could arrange it, and standard attribution
inspection would not notice.

the method is an auxiliary penalty during ppo training on the divergence
between pi(.|s) and pi(.|s') where the target feature is resampled from the
batch marginal. that is the same interventional perturbation shapley
attribution measures, so the penalty attacks the attribution directly.

THE CONTROL IS THE POINT. the experiment is run on two corpora:

  real market        the target feature dominates the explanation but the
                     agent has no measurable edge. prediction: steering
                     succeeds, since nothing load-bearing is being removed.

  learnable synthetic  the target feature is the planted signal and is
                     genuinely necessary. prediction: steering FAILS, because
                     removing it costs return.

if both hold, the finding is sharp and falsifiable: **explanations are steerable
exactly when they are not tracking anything real.** that ties directly to the
null test, which says the real agent's explanation is not tracking anything.

if steering also succeeds on the synthetic corpus, the result is weaker and
different: it would mean the penalty is defeating the attribution method rather
than changing what the agent uses, and the report must say so.

usage:
    python scripts/steer_explanation.py --corpus data/corpus/corpus_candles_60s_spot.npz
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
from nano_rl.env.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    SIGNAL_OBS_IDX,
    feature_index,
    fit_normalizer,
)
from nano_rl.env.synthetic import make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402
from nano_rl.explain.trajectory import explain_behaviour  # noqa: E402
from nano_rl.metrics import (  # noqa: E402
    cluster_bootstrap_p_value,
    paired_bootstrap_p_value,
)


# the penalty the paper reports, fixed before evaluation.
REPORTED_COEF = 20.0


def banner(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}", flush=True)


def global_attribution(
    agent: PPOAgent, env: BinaryMarketEnv, background: np.ndarray,
    n_states: int = 25, n_permutations: int = 50, seed: int = 0,
) -> np.ndarray:
    """mean absolute per-decision attribution over sampled states."""
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
        att, _ = explain_behaviour(
            agent, obs, background, n_permutations=n_permutations, seed=seed + i
        )
        acc += np.abs(att.values)
    return acc / n_states


def run_condition(
    train_batch, eval_batch, norm, target: int, coef: float,
    updates: int, seeds: int, n_states: int, base_seed: int,
) -> dict:
    """train `seeds` agents at one penalty strength; measure both quantities."""
    returns, shares = [], []
    all_pnl = []

    for s in range(seeds):
        env = BinaryMarketEnv(train_batch, max_position=100.0, normalizer=norm)
        env.reset(seed=base_seed + s)
        cfg = PPOConfig(
            seed=base_seed + s,
            invariance_feature=target if coef > 0 else None,
            invariance_coef=coef,
        )
        agent = PPOAgent(N_FEATURES, 3, cfg)
        agent.train(env, n_updates=updates, verbose=False)

        ev = BinaryMarketEnv(
            eval_batch, max_position=100.0, normalizer=norm,
            random_episode_order=False,
        )
        res = agent.evaluate(ev, n_episodes=min(500, len(eval_batch)))
        returns.append(float(res["returns"].mean()))
        all_pnl.append(res["returns"])

        roll = VectorizedRollout(eval_batch, normalizer=norm, max_position=100.0)
        bg = build_background(roll, n_samples=192, seed=base_seed + s)
        att = global_attribution(
            agent, ev, bg, n_states=n_states, seed=base_seed + s
        )
        total = att.sum()
        shares.append(float(att[target] / total) if total > 0 else 0.0)

    return {
        "coef": coef,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "target_share_mean": float(np.mean(shares)),
        "target_share_std": float(np.std(shares)),
        "pnl": all_pnl,
    }


def run_corpus(
    name: str, train_batch, eval_batch, norm, target: int,
    coefs: list[float], updates: int, seeds: int, n_states: int,
    base_seed: int, prediction: str,
) -> list[dict]:
    banner(f"{name}   target feature: `{FEATURE_NAMES[target]}`")
    print(f"  prediction: {prediction}\n")
    print(f"  {'penalty':>9} {'return':>18} {'attribution share':>20} "
          f"{'p pooled':>10}{'p cluster':>11}")

    rows = []
    baseline_pnl = None
    for c in coefs:
        r = run_condition(
            train_batch, eval_batch, norm, target, c,
            updates, seeds, n_states, base_seed,
        )
        if baseline_pnl is None:
            baseline_pnl = r["pnl"]
            p = float("nan")
        else:
            # one paired bootstrap over every matched episode from every seed.
            #
            # this previously took the median of the per-seed p-values, which
            # is not a combined test: the median of k p-values has no defined
            # size, and reporting it as "the" p-value understated how much
            # seed-to-seed variation there is. pooling the paired differences
            # keeps the pairing that makes the test worth running (both agents
            # see the same episodes) and gives one quantity with a meaning.
            pooled_a = np.concatenate([np.asarray(x) for x in r["pnl"]])
            pooled_b = np.concatenate([np.asarray(x) for x in baseline_pnl])
            p = paired_bootstrap_p_value(pooled_a, pooled_b)
            # keep the spread so the reader can see the seeds disagree
            per_seed = [
                paired_bootstrap_p_value(a, b)
                for a, b in zip(r["pnl"], baseline_pnl)
            ]
            r["p_per_seed"] = [float(x) for x in per_seed]
            # the seed, not the episode, is the unit of independent
            # replication: one trained policy generates every episode in a
            # seed. the pooled test above answers how precisely we know these
            # three agents' mean, the cluster test answers whether the effect
            # would reappear in a fourth training run. where the seeds
            # disagree the two come apart, and the paper quotes the
            # conservative one.
            r["p_cluster"] = float(
                cluster_bootstrap_p_value(r["pnl"], baseline_pnl)
            )
        r["p_vs_baseline"] = p
        r.pop("pnl")
        rows.append(r)

        print(
            f"  {c:>9.1f} {r['return_mean']:>+10.2f} +/-{r['return_std']:<5.2f} "
            f"{r['target_share_mean']:>14.1%} +/-{r['target_share_std']:<4.1%} "
            f"{p:>10.3f}{r.get('p_cluster', float('nan')):>11.3f}",
            flush=True,
        )

    base = rows[0]
    # report a PRE-SPECIFIED penalty rather than the one with the strongest
    # suppression. picking the best of four on the same evaluation that then
    # reports it is a selection effect, and the paper quotes this number.
    chosen = next((r for r in rows[1:] if r["coef"] == REPORTED_COEF), None)
    if chosen is None:
        chosen = rows[-1] if len(rows) > 1 else base
    drop = 1.0 - (chosen["target_share_mean"] / max(base["target_share_mean"], 1e-9))
    # a non-significant difference is not evidence of equivalence. this records
    # only that no difference was detected at this power, which is the weaker
    # and supportable claim.
    # the conservative test governs. a pooled p-value that treats episodes
    # from one agent as independent replicates can read as significant on the
    # strength of a single seed, which is what happens here.
    p_pooled = chosen["p_vs_baseline"]
    p_cluster = chosen.get("p_cluster", float("nan"))
    undetected = (p_cluster >= 0.05) if p_cluster == p_cluster else (p_pooled >= 0.05)

    print(f"\n  attribution to `{FEATURE_NAMES[target]}` fell "
          f"{base['target_share_mean']:.1%} -> {chosen['target_share_mean']:.1%} "
          f"({drop:.0%} reduction) at the pre-specified coef {REPORTED_COEF:g}")
    print(f"  return difference vs baseline: "
          f"{'NOT DETECTED' if undetected else 'DETECTED'} "
          f"(cluster p = {p_cluster:.3f}, pooled p = {p_pooled:.3f}; "
          f"this is not an equivalence test)")
    print(f"  per-seed p: {[round(x, 4) for x in chosen.get('p_per_seed', [])]}")
    print(f"  return {base['return_mean']:+.3f} -> {chosen['return_mean']:+.3f} "
          f"(change {chosen['return_mean'] - base['return_mean']:+.3f})")
    verdict = ("STEERABLE WITHOUT DETECTED COST" if (drop > 0.5 and undetected)
               else "STEERING HAS A DETECTED COST")
    print(f"  -> {verdict}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--coefs", type=float, nargs="+",
                    default=[0.0, 1.0, 5.0, 20.0])
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-states", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- the real market
    batch = EpisodeBatch.load(args.corpus)
    split = walk_forward_split(batch)
    target_real = feature_index("time_to_expiry_frac")

    real_rows = run_corpus(
        "REAL MARKET (agent has no measurable edge)",
        split.train, split.test, split.normalizer, target_real,
        args.coefs, args.updates, args.seeds, args.n_states, args.seed,
        "steering should SUCCEED: nothing load-bearing is being removed",
    )

    # ---------------------------------------------------------- the control
    lb = make_learnable_corpus(n_episodes=1500, seed=args.seed)
    ln = fit_normalizer(lb)

    synth_rows = run_corpus(
        "LEARNABLE SYNTHETIC (the target feature IS the signal)",
        lb, lb, ln, SIGNAL_OBS_IDX,
        args.coefs, args.updates, args.seeds, args.n_states, args.seed,
        "steering should FAIL: removing the signal must cost return",
    )

    # -------------------------------------------------------------- verdict
    banner("WHAT THIS MEANS")

    def summarise(rows):
        # same pre-specified penalty as above, and its own p-value. the
        # previous version selected the strongest suppression and then paired
        # it with a p-value taken from elsewhere, which is how the paper came
        # to quote an attribution from one run beside a return from another.
        base = rows[0]
        pick = next((r for r in rows[1:] if r["coef"] == REPORTED_COEF), rows[-1])
        drop = 1.0 - (pick["target_share_mean"] / max(base["target_share_mean"], 1e-9))
        # the CLUSTER p governs here, the same quantity the per-corpus block
        # above decides on. an earlier version returned the pooled p while the
        # block above used the cluster p, so the two halves of this script drew
        # opposite conclusions from one run: the market was reported
        # "steerable without detected cost" and then, four lines later,
        # "steering did not succeed even on the real market".
        p_c = pick.get("p_cluster", float("nan"))
        p_used = p_c if p_c == p_c else pick["p_vs_baseline"]
        return (drop, p_used, pick["p_vs_baseline"],
                base["return_mean"], pick["return_mean"])

    rd, rp, rp_pool, rb0, rb1 = summarise(real_rows)
    sd, sp, sp_pool, sb0, sb1 = summarise(synth_rows)

    print(f"  {'corpus':<26}{'attr drop':>11}{'return':>22}"
          f"{'p cluster':>11}{'p pooled':>10}")
    print(f"  {'real market':<26}{rd:>10.0%}  {rb0:>+8.2f} -> {rb1:>+8.2f}"
          f"{rp:>11.3f}{rp_pool:>10.3f}")
    print(f"  {'learnable synthetic':<26}{sd:>10.0%}  {sb0:>+8.2f} -> {sb1:>+8.2f}"
          f"{sp:>11.3f}{sp_pool:>10.3f}")

    # "no detected difference", not "performance preserved"
    steerable_real = rd > 0.5 and rp >= 0.05
    steerable_synth = sd > 0.5 and sp >= 0.05

    print()
    if steerable_real and not steerable_synth:
        print("  the explanation is steerable on the real market and NOT on the")
        print("  corpus where the feature genuinely matters. so explanations are")
        print("  steerable exactly where they are not tracking anything real,")
        print("  which is precisely what the null test says of the real agent.")
    elif steerable_real and steerable_synth:
        print("  steering succeeded on BOTH, including where the feature is")
        print("  genuinely load-bearing. that is a weaker and different result:")
        print("  the penalty is defeating the attribution method rather than")
        print("  changing what the agent relies on. reported as such.")
    elif not steerable_real:
        print("  steering did not succeed even on the real market. the")
        print("  explanation resists being changed at fixed performance, which")
        print("  is evidence it tracks something the penalty cannot remove.")

    plots.steering(
        {"real market": real_rows, "learnable synthetic": synth_rows},
        out / "explanation_steering.png",
        target_names={
            "real market": FEATURE_NAMES[target_real],
            "learnable synthetic": FEATURE_NAMES[SIGNAL_OBS_IDX],
        },
    )

    (out / "steering.json").write_text(json.dumps(
        {"real_market": real_rows, "learnable_synthetic": synth_rows,
         "steerable_real": bool(steerable_real),
         "steerable_synthetic": bool(steerable_synth),
         "decided_on": "p_cluster",
         "p_cluster_real": float(rp), "p_pooled_real": float(rp_pool),
         "p_cluster_synthetic": float(sp), "p_pooled_synthetic": float(sp_pool)},
        indent=2,
    ))
    print(f"\nwrote {out}/steering.json")


if __name__ == "__main__":
    main()
