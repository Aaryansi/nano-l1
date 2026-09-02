"""the canonical parameter-randomization sanity check, run on our own agents.

this is a different experiment from the parameter null in
scripts/generalize_gym.py, and the distinction is the point of running it.

  - the parameter NULL builds a reference distribution: it randomizes weights
    many times, records the attribution span of each random policy, and asks
    where the trained agent's span falls in that distribution.

  - the canonical check of adebayo et al. (2018) compares ONE explanation
    against ONE degraded copy of itself: randomize the trained network's layers
    progressively and measure how much the explanation moves. if it does not
    move, the explanation does not depend on what the network learned.

the paper previously described the first as "the established sanity check",
which invites the objection that we tested something adebayo et al. did not
propose and then reported it as underpowered. so we run the actual check here.

the question worth answering is not whether the check works, but whether it
separates our two agents. the market agent's explanation is empty in the sense
that its outcome-attribution span is indistinguishable from agents trained
where there is provably nothing to learn. the planted-signal agent's is not.
if the canonical check moves the same amount on both, then passing it is not
evidence that there is anything to explain, which is the paper's thesis stated
in the incumbent's own terms.

randomization follows adebayo et al.: cascading (top-down, accumulating) and
independent (one layer at a time from the trained weights). the similarity
metric for a feature-vector attribution is spearman rank correlation on the
absolute values, the analogue of their rank correlation on saliency maps,
reported alongside cosine similarity.

usage:
    python scripts/parameter_randomization.py --corpus <path> --runs runs/ppo \
        --out reports
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
from nano_rl.env.synthetic import make_learnable_corpus  # noqa: E402
from nano_rl.explain.rollout import VectorizedRollout, build_background  # noqa: E402

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
                "rank_corr_mean": float(np.nanmean(rhos)),
                "rank_corr_std": float(np.nanstd(rhos)),
                "cosine_mean": float(np.nanmean(coss)),
                "cosine_std": float(np.nanstd(coss)),
            }
            out[mode].append(row)
            print(f"  {mode:<12} {layer:<12} "
                  f"rho {row['rank_corr_mean']:>+7.3f} +/- {row['rank_corr_std']:.3f}   "
                  f"cos {row['cosine_mean']:>+7.3f}")

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

    planted = make_learnable_corpus(n_episodes=1500, seed=args.seed)
    pnorm = fit_normalizer(planted)
    penv_train = BinaryMarketEnv(planted, max_position=100.0, normalizer=pnorm)
    penv_train.reset(seed=args.seed)
    pagent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=args.seed))
    pagent.train(penv_train, n_updates=args.updates, verbose=False)
    penv = BinaryMarketEnv(planted, max_position=100.0, normalizer=pnorm,
                           random_episode_order=False)
    proll = VectorizedRollout(planted, normalizer=pnorm, max_position=100.0)
    pbg = build_background(proll, n_samples=192, seed=args.seed)

    result = {
        "n_states": args.n_states,
        "seeds": args.seeds,
        "market": run_agent("market agent (empty explanation)", market, menv,
                            mbg, args.n_states, args.seeds, args.seed),
        "planted": run_agent("planted-signal agent (informative)", pagent, penv,
                             pbg, args.n_states, args.seeds, args.seed),
    }

    # the comparison the paper needs: does the check separate the two agents?
    def deepest(d):
        return d["cascading"][-1]["rank_corr_mean"]

    m, p = deepest(result["market"]), deepest(result["planted"])
    result["separates_agents"] = bool(abs(m - p) > 0.3)
    result["fully_randomized_rank_corr"] = {"market": m, "planted": p}

    banner("verdict")
    print(f"  fully randomized, market  rho = {m:+.3f}")
    print(f"  fully randomized, planted rho = {p:+.3f}")
    print(f"  separates the two agents: {result['separates_agents']}")
    print("\n  the canonical check asks whether an explanation depends on the")
    print("  learned weights. it does not ask whether the thing explained is")
    print("  worth explaining, and these two agents differ on the second.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parameter_randomization.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out / 'parameter_randomization.json'}")


if __name__ == "__main__":
    main()
