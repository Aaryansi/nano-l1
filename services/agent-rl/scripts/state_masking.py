"""Compare observation replacement while preserving declared state structure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch
from nano_rl.env.features import N_FEATURES, fit_normalizer
from nano_rl.env.synthetic import make_learnable_corpus, make_null_corpus
from nano_rl.explain.interventions import MODES, intervention_rollout
from nano_rl.explain.rollout import VectorizedRollout, build_background, greedy_policy
from nano_rl.metrics import cluster_bootstrap_ci


def measure_case(name, units, evaluation, out_raw, market=False):
    """All units share evaluation episodes; masking is fixed within each seed."""
    values = {"unmodified": []} | {mode: [] for mode in MODES}
    rows = []
    for seed, agent, norm in units:
        roll = VectorizedRollout(evaluation, normalizer=norm, max_position=100.)
        background = build_background(roll, n_samples=256, seed=seed)
        policy = greedy_policy(agent)
        observed = roll.run(policy)["returns"]
        values["unmodified"].append(observed)
        row = {"seed": seed, "masking_seed": seed, "background_samples": len(background),
               "unmodified_return": float(observed.mean()), "spans": {}}
        for mode in MODES:
            replaced = intervention_rollout(policy, roll, background, mode, seed)["returns"]
            values[mode].append(replaced)
            row["spans"][mode] = float((observed - replaced).mean())
        rows.append(row)
    blocks = (np.floor(evaluation.open_epoch / 86400).astype(int) if market
              else np.arange(len(evaluation)))
    summaries = {}
    for mode in MODES:
        delta = np.asarray(values["unmodified"]) - np.asarray(values[mode])
        summaries[mode] = {"mean_span": float(delta.mean()),
                           "per_seed_spans": delta.mean(axis=1).tolist(),
                           "ci95": list(cluster_bootstrap_ci(values["unmodified"], values[mode],
                                                             block_ids=blocks)),
                           "uncertainty": "crossed seeds and shared UTC days" if market else
                                          "crossed seeds and shared independent episodes"}
    for mode, arrays in values.items():
        out_raw[f"{name}__{mode}"] = np.asarray(arrays)
    out_raw[f"{name}__open_epoch"] = evaluation.open_epoch
    return {"case": name, "n_episodes": len(evaluation), "per_seed": rows,
            "interventions": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=1200)
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("at least two seeds are required for the declared seed uncertainty")
    checkpoints = sorted(args.runs.glob("seed*.pt"))
    if len(checkpoints) != args.seeds:
        parser.error("checkpoint count must match the declared seed count")
    split = walk_forward_split(EpisodeBatch.load(args.corpus))
    raw, cases, units = {}, [], []
    for path in checkpoints:
        seed = int(path.stem.removeprefix("seed"))
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
        agent.load(str(path))
        units.append((seed, agent, split.normalizer))
    cases.append(measure_case("market", units, split.test, raw, market=True))
    for index, (name, factory) in enumerate((("null", make_null_corpus),
                                            ("signal", make_learnable_corpus))):
        evaluation_seed = 700000 + index * 1000
        evaluation = factory(n_episodes=args.episodes, seed=evaluation_seed)
        units, train_seeds = [], []
        for seed in range(args.seeds):
            train_seed = 600000 + index * 1000 + seed
            training = factory(n_episodes=args.episodes, seed=train_seed)
            norm = fit_normalizer(training)
            agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seed))
            env = BinaryMarketEnv(training, normalizer=norm, max_position=100.)
            env.reset(seed=seed)
            agent.train(env, n_updates=args.updates, verbose=False)
            env.close()
            units.append((seed, agent, norm))
            train_seeds.append(train_seed)
        case = measure_case(name, units, evaluation, raw)
        case.update(training_corpus_seeds=train_seeds, evaluation_corpus_seed=evaluation_seed,
                    training_updates=args.updates)
        cases.append(case)
    result = {"cases": cases, "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
              "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints},
              "limitations": ["fixed masking draws; intervals omit intervention-seed variation",
                              "market replacements retain inventory and time; inventory inputs can reveal market data",
                              "donor sequence support does not establish joint recipient-state consistency",
                              "span is intervention sensitivity, not attribution accuracy or market efficiency"]}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "state_masking_returns.npz", **raw)
    (args.out / "state_masking.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(case["case"], {mode: row["mean_span"] for mode, row in case["interventions"].items()}, flush=True)


if __name__ == "__main__":
    main()
