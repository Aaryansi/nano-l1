"""why does ppo collapse to always-FLAT on a corpus with real signal?

the environment is known good: tabular q reaches 98.2% of optimum on the same
corpus. so the fault is in the agent or its inputs.

hypotheses:
  H1  input scale. the observation is fed raw. volume_rate is log1p(10000)
      = 9.21, a large constant. a tanh trunk saturates on inputs that size,
      gradients vanish, and the network cannot learn regardless of signal.
  H2  entropy collapse. random trading loses money because of fees, so FLAT
      looks best early. if entropy decays before the policy learns to
      CONDITION on the signal, it locks in abstention.
  H3  advantage propagation. reward is -2.75 at entry, zero for twelve steps,
      then +/-50 at settlement. with gae lambda 0.95 over 13 steps only
      0.95^13 = 0.51 of the terminal signal reaches the entry decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_rl.agents.networks import ActorCritic  # noqa: E402
from nano_rl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from nano_rl.env.binary_market import BinaryMarketEnv  # noqa: E402
from nano_rl.env.features import FEATURE_NAMES, N_FEATURES  # noqa: E402
from nano_rl.env.synthetic import make_learnable_corpus, signal_policy_return  # noqa: E402


def main() -> None:
    batch = make_learnable_corpus(n_episodes=2000, seed=0)
    bench = signal_policy_return(batch, max_position=100.0)

    # ------------------------------------------------------------------ H1
    print("=" * 74)
    print("H1: what does the raw observation actually look like?")
    print("=" * 74)
    env = BinaryMarketEnv(batch, max_position=100.0, random_episode_order=False)
    obs_samples = []
    for ep in range(200):
        o, _ = env.reset(options={"episode": ep})
        obs_samples.append(o)
        for _ in range(3):
            o, _, d, _, _ = env.step(1)
            obs_samples.append(o)
            if d:
                break
    obs_arr = np.array(obs_samples)

    print(f"  {'feature':<24} {'mean':>10} {'std':>9} {'min':>9} {'max':>9}")
    for i, nm in enumerate(FEATURE_NAMES):
        c = obs_arr[:, i]
        flag = "  <-- LARGE" if abs(c).max() > 5 else ""
        print(f"  {nm:<24} {c.mean():>10.3f} {c.std():>9.3f} "
              f"{c.min():>9.3f} {c.max():>9.3f}{flag}")

    # how saturated is the first tanh layer at init?
    net = ActorCritic(N_FEATURES, 3)
    with torch.no_grad():
        x = torch.as_tensor(obs_arr, dtype=torch.float32)
        pre = net.trunk[0](x)
        post = torch.tanh(pre)
        sat = (post.abs() > 0.95).float().mean()
    print(f"\n  fraction of first-layer tanh units saturated (|out| > 0.95): {sat:.3f}")
    print("  a saturated unit has gradient ~0, so the trunk cannot learn.")

    # ------------------------------------------------------------------ fix
    print("\n" + "=" * 74)
    print("does normalising the observation fix it?")
    print("=" * 74)

    from nano_rl.env.features import FeatureNormalizer

    feats = batch.market_features()
    norm = FeatureNormalizer().fit(feats.reshape(-1, feats.shape[-1]))

    nenv = BinaryMarketEnv(batch, max_position=100.0, random_episode_order=False)
    nenv.normalizer = norm
    nobs = []
    for ep in range(200):
        o, _ = nenv.reset(options={"episode": ep})
        nobs.append(o)
    nobs_arr = np.array(nobs)
    with torch.no_grad():
        post2 = torch.tanh(net.trunk[0](torch.as_tensor(nobs_arr, dtype=torch.float32)))
        sat2 = (post2.abs() > 0.95).float().mean()
    print(f"  saturated fraction after normalisation: {sat2:.3f} (was {sat:.3f})")

    # ---------------------------------------------------------------- trials
    print("\n" + "=" * 74)
    print("ppo under each remedy (30 updates each, benchmark "
          f"{bench:.2f})")
    print("=" * 74)
    print(f"  {'variant':<34} {'return':>9} {'trades':>8} {'entropy':>9} {'ev':>7}")

    trials = [
        ("raw obs, entropy 0.01 (baseline)", None, 0.01),
        ("normalised obs, entropy 0.01", norm, 0.01),
        ("normalised obs, entropy 0.05", norm, 0.05),
        ("raw obs, entropy 0.05", None, 0.05),
    ]

    for label, normalizer, ent in trials:
        e = BinaryMarketEnv(batch, max_position=100.0, normalizer=normalizer)
        agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=0, entropy_coef=ent))
        log = agent.train(e, n_updates=30, verbose=False)
        ev = BinaryMarketEnv(
            batch, max_position=100.0, normalizer=normalizer, random_episode_order=False
        )
        r = agent.evaluate(ev, n_episodes=400)
        print(f"  {label:<34} {r['returns'].mean():>9.2f} "
              f"{r['trades'].mean():>8.2f} {log.entropy[-1]:>9.3f} "
              f"{log.explained_var[-1]:>7.3f}")


if __name__ == "__main__":
    main()
