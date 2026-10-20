"""Independent PPO training with an actor-only export for attribution."""
from __future__ import annotations

import torch
import copy
import numpy as np

from nano_rl.agents.networks import ActorCritic


def export_actor(policy, hidden=64):
    """Copy the categorical actor without consuming the training RNG stream.

    The exported value head is intentionally zero and is not a trained critic.
    Control-task attribution uses action probabilities and episode returns only.
    """
    with torch.random.fork_rng(devices=[]):
        actor = ActorCritic(int(policy.observation_space.shape[0]),
                            int(policy.action_space.n), hidden)
        actor.trunk.load_state_dict(policy.mlp_extractor.policy_net.state_dict())
        actor.policy_head.load_state_dict(policy.action_net.state_dict())
        with torch.no_grad():
            actor.value_head.weight.zero_()
            actor.value_head.bias.zero_()
    return actor


def train_independent_ppo(env, cfg, total_steps, checkpoint_fractions=()):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    checkpoints = []
    wanted = sorted(checkpoint_fractions)

    class SnapshotCallback(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_start(self):
            # The previous rollout's optimizer update has completed here.
            while wanted and self.num_timesteps >= wanted[0] * total_steps:
                fraction = wanted.pop(0)
                checkpoints.append((fraction, export_actor(self.model.policy, cfg.hidden)))

    def copy_env(index):
        cloned = copy.deepcopy(env)
        wrapper = cloned
        while hasattr(wrapper, "env"):
            if "_rng" in vars(wrapper):
                wrapper._rng = np.random.default_rng(cfg.seed + 200000 + index)
            wrapper = wrapper.env
        return cloned

    vec = DummyVecEnv([lambda i=i: copy_env(i) for i in range(cfg.n_envs)])
    model = PPO("MlpPolicy", vec, seed=cfg.seed, device="cpu", verbose=0,
                learning_rate=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_coef, ent_coef=cfg.entropy_coef,
                vf_coef=cfg.value_coef, max_grad_norm=cfg.max_grad_norm,
                n_steps=max(2, cfg.steps_per_batch // cfg.n_envs), n_epochs=cfg.update_epochs,
                batch_size=cfg.minibatch_size,
                policy_kwargs={"net_arch": {"pi": [cfg.hidden, cfg.hidden],
                                            "vf": [cfg.hidden, cfg.hidden]}})
    model.learn(total_timesteps=total_steps, callback=SnapshotCallback())
    final = export_actor(model.policy, cfg.hidden)
    while wanted and model.num_timesteps >= wanted[0] * total_steps:
        checkpoints.append((wanted.pop(0), export_actor(model.policy, cfg.hidden)))
    vec.close()
    return final, checkpoints
