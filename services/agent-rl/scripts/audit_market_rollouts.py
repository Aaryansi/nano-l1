"""Audit every held-out trajectory of the supplied market checkpoints.

The scalar environment and vector replay independently construct observations
and account for trades. Compare all acted-on observations, deterministic
actions, episode returns, fees, trade counts and final positions, then compare
both implementations with the saved primary evaluation arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.data.splits import walk_forward_split
from nano_rl.env.binary_market import BinaryMarketEnv, EpisodeBatch
from nano_rl.env.costs import CostModel
from nano_rl.env.features import N_FEATURES
from nano_rl.explain.rollout import VectorizedRollout, greedy_policy


def require(condition, context):
    if not condition:
        raise ValueError(f"market trajectory audit failed: {context}")


def audit(corpus: Path, run: Path) -> dict:
    split = walk_forward_split(EpisodeBatch.load(str(corpus)))
    rows = []
    with np.load(run / "reports/evaluation_returns.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["open_epoch"], split.test.open_epoch), "episode identities")
        for directory, costs in (("ppo", True), ("ppo_nofric", False)):
            checkpoints = sorted((run / "checkpoints" / directory).glob("seed*.pt"))
            require(checkpoints and len(checkpoints) == saved[f"{directory}_pnl"].shape[0],
                    f"{directory} checkpoint count")
            for index, checkpoint in enumerate(checkpoints):
                agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0)).load(str(checkpoint))
                policy = greedy_policy(agent)
                observations, actions = [], []

                def record(obs):
                    selected = policy(obs)
                    observations.append(obs.copy())
                    actions.append(selected.copy())
                    return selected

                vector = VectorizedRollout(split.test, normalizer=split.normalizer,
                                          max_position=100., costs_enabled=costs).run(record)
                env = BinaryMarketEnv(split.test, normalizer=split.normalizer,
                                      cost_model=CostModel(enabled=costs), max_position=100.,
                                      random_episode_order=False)
                scalar = {key: [] for key in vector}
                for episode in range(len(split.test)):
                    obs, _ = env.reset(options={"episode": episode})
                    total = 0.
                    for step in range(split.test.n_steps):
                        context = f"{checkpoint.name}, episode {episode}, step {step}"
                        require(np.array_equal(obs, observations[step][episode]), context + ": observation")
                        with torch.no_grad():
                            selected, _, _ = agent.net.act(torch.as_tensor(obs).unsqueeze(0),
                                                           deterministic=True)
                        action = int(selected.item())
                        require(action == actions[step][episode], context + ": action")
                        obs, reward, done, truncated, info = env.step(action)
                        total += reward
                        require(done == (step == split.test.n_steps - 1) and not truncated,
                                context + ": termination")
                    for key, value in (("returns", total), ("trades", info["trades"]),
                                       ("fees", info["fees"]), ("final_position", info["position"])):
                        scalar[key].append(value)
                errors = {}
                for key in vector:
                    values = np.asarray(scalar[key])
                    errors[key] = float(np.max(np.abs(values - vector[key])))
                    require(np.isfinite(values).all() and np.isfinite(vector[key]).all()
                            and np.allclose(values, vector[key], rtol=0, atol=1e-10),
                            f"{checkpoint.name}: {key}")
                for suffix, key in (("pnl", "returns"), ("trades", "trades"), ("fees", "fees")):
                    require(np.allclose(vector[key], saved[f"{directory}_{suffix}"][index],
                                        rtol=0, atol=1e-9), f"{checkpoint.name}: saved {suffix}")
                row = {
                    "checkpoint": f"{directory}/{checkpoint.name}",
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "episodes": len(split.test), "steps": split.test.n_steps,
                    "acted_on_observations_checked": len(split.test) * split.test.n_steps,
                    "actions_checked": len(split.test) * split.test.n_steps,
                    "maximum_observation_error": 0., "maximum_accounting_errors": errors,
                    "matches_saved_evaluation": True,
                }
                rows.append(row)
                print(f"verified {row['checkpoint']}: {row['actions_checked']} actions", flush=True)
    return {
        "schema_version": 1,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "all acted-on observations and greedy actions; all episode accounts; all supplied market checkpoints",
        "checkpoints": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = audit(args.corpus, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
