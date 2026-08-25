"""ppo with gae, written from scratch.

why ppo rather than dqn, given the spec allowed either:

  1. the correct policy on this problem is mostly "do not trade", and
     scripts/diagnose_null.py measured a max-operator method getting that
     wrong in the dangerous direction. tabular q learned SHORT at -1.258
     against a true -2.750 and FLAT at -0.701 against a true 0.000, so it
     overestimated trading and underestimated abstention. a policy gradient
     has no max operator and does not inherit that bias.

  2. ppo carries an explicit value head. docs/MDP.md section 1.2 turns V(s)
     into a falsifiable object, because every episode resolves to a known 0/1
     and predicted values can be checked against realised frequencies. that
     calibration figure is a phase-5c deliverable, so a critic is required
     rather than optional.

  3. on a near-zero-signal problem, ppo's trust region limits how far a noisy
     advantage estimate can move the policy in one step.

references drawn on, not copied: schulman et al. 2017 for the clipped
surrogate, schulman et al. 2016 for gae, and the implementation-details survey
by engstrom et al. 2020 / huang et al. 2022 for advantage normalisation,
orthogonal init, and gradient clipping. the code here is written directly
against those descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from nano_rl.agents.networks import ActorCritic
from nano_rl.env.binary_market import BinaryMarketEnv


@dataclass
class PPOConfig:
    """hyperparameters. defaults are the standard ppo settings except where
    noted, since tuning on a no-signal problem mostly fits noise."""

    lr: float = 3e-4
    gamma: float = 1.0  # finite-horizon episodes; see docs/MDP.md section 7.1
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    episodes_per_batch: int = 64
    update_epochs: int = 4
    minibatch_size: int = 256

    # anneal the learning rate to zero over training, which is standard and
    # matters more than usual when the gradient signal is mostly noise.
    anneal_lr: bool = True

    hidden: int = 64
    seed: int = 0
    device: str = "cpu"


@dataclass
class PPOLog:
    """per-update diagnostics. everything a reviewer needs to see whether
    training did anything, including the failure modes."""

    updates: list[int] = field(default_factory=list)
    mean_return: list[float] = field(default_factory=list)
    policy_loss: list[float] = field(default_factory=list)
    value_loss: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    approx_kl: list[float] = field(default_factory=list)
    clip_frac: list[float] = field(default_factory=list)
    explained_var: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    action_freq: list[tuple[float, float, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {
            "updates": self.updates,
            "mean_return": self.mean_return,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_frac": self.clip_frac,
            "explained_var": self.explained_var,
            "lr": self.lr,
            "action_freq": self.action_freq,
        }


class RolloutBuffer:
    """stores complete episodes and computes gae advantages.

    episodes are stored whole rather than as a fixed-length window. this env
    always terminates at a true terminal state after a known number of steps,
    so there is never a bootstrap-through-truncation case to handle, and
    getting that wrong is a classic silent bug.
    """

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.episode_starts: list[int] = []

    def start_episode(self) -> None:
        self.episode_starts.append(len(self.obs))

    def add(
        self, obs: np.ndarray, action: int, log_prob: float, value: float, reward: float
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def __len__(self) -> int:
        return len(self.obs)

    def compute_gae(self, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """generalised advantage estimation, per episode.

        the terminal value is exactly zero rather than a bootstrap, because
        every episode ends at a genuine terminal state where the contract has
        settled and there is no future.
        """
        rewards = np.asarray(self.rewards, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        advantages = np.zeros_like(rewards)

        bounds = self.episode_starts + [len(self.obs)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            gae = 0.0
            for t in reversed(range(a, b)):
                next_value = values[t + 1] if t + 1 < b else 0.0
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * lam * gae
                advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def episode_returns(self) -> np.ndarray:
        """undiscounted total reward per episode, for logging."""
        rewards = np.asarray(self.rewards)
        bounds = self.episode_starts + [len(self.obs)]
        return np.array([rewards[a:b].sum() for a, b in zip(bounds[:-1], bounds[1:])])


class PPOAgent:
    """ppo with a categorical policy over target positions."""

    def __init__(self, n_features: int, n_actions: int, cfg: PPOConfig | None = None) -> None:
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.device = torch.device(self.cfg.device)
        self.net = ActorCritic(n_features, n_actions, self.cfg.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.n_actions = n_actions
        self._rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------- rollout

    def collect(self, env: BinaryMarketEnv, n_episodes: int) -> RolloutBuffer:
        """run the current policy for `n_episodes` complete episodes."""
        buf = RolloutBuffer()

        for _ in range(n_episodes):
            obs, _ = env.reset()
            buf.start_episode()
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, log_prob, value = self.net.act(obs_t.unsqueeze(0))
                a = int(action.item())
                next_obs, reward, done, _, _ = env.step(a)
                buf.add(obs, a, float(log_prob.item()), float(value.item()), float(reward))
                obs = next_obs
                if done:
                    break

        return buf

    # -------------------------------------------------------------- update

    def update(self, buf: RolloutBuffer) -> dict[str, float]:
        """one ppo update over the collected batch."""
        cfg = self.cfg
        advantages, returns = buf.compute_gae(cfg.gamma, cfg.gae_lambda)

        obs = torch.as_tensor(np.asarray(buf.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buf.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            buf.log_probs, dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(buf.values, dtype=torch.float32, device=self.device)

        n = len(buf)
        idx = np.arange(n)

        pol_losses, val_losses, entropies, kls, clip_fracs = [], [], [], [], []

        for _ in range(cfg.update_epochs):
            self._rng.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                new_log_probs, entropy, values = self.net.evaluate_actions(
                    obs[mb_t], actions[mb_t]
                )

                log_ratio = new_log_probs - old_log_probs[mb_t]
                ratio = log_ratio.exp()

                # advantage normalisation, per minibatch. important here
                # because raw advantages are in dollars and swing by ~50.
                mb_adv = adv_t[mb_t]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # clipped surrogate objective
                pg_1 = -mb_adv * ratio
                pg_2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_1, pg_2).mean()

                # clipped value loss, mirroring the policy clip
                v_unclipped = (values - ret_t[mb_t]) ** 2
                v_clipped = old_values[mb_t] + torch.clamp(
                    values - old_values[mb_t], -cfg.clip_coef, cfg.clip_coef
                )
                v_clipped = (v_clipped - ret_t[mb_t]) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # schulman's low-variance approximate kl
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                pol_losses.append(float(policy_loss.item()))
                val_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))

        # explained variance of the critic: 1 means it predicts returns
        # perfectly, 0 means it does no better than predicting the mean, and
        # negative means it is worse than that.
        y_true = returns
        y_pred = np.asarray(buf.values)
        var_y = float(np.var(y_true))
        explained = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        counts = np.bincount(np.asarray(buf.actions), minlength=self.n_actions)
        freq = tuple((counts / max(counts.sum(), 1)).tolist())

        return {
            "policy_loss": float(np.mean(pol_losses)),
            "value_loss": float(np.mean(val_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_var": explained,
            "action_freq": freq,
        }

    # --------------------------------------------------------------- train

    def train(
        self,
        env: BinaryMarketEnv,
        n_updates: int,
        log_every: int = 10,
        verbose: bool = True,
    ) -> PPOLog:
        """collect-and-update loop."""
        log = PPOLog()

        for update in range(n_updates):
            if self.cfg.anneal_lr:
                frac = 1.0 - update / max(n_updates, 1)
                new_lr = frac * self.cfg.lr
                for group in self.optimizer.param_groups:
                    group["lr"] = new_lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            buf = self.collect(env, self.cfg.episodes_per_batch)
            stats = self.update(buf)
            ep_rets = buf.episode_returns()

            log.updates.append(update)
            log.mean_return.append(float(ep_rets.mean()))
            log.policy_loss.append(stats["policy_loss"])
            log.value_loss.append(stats["value_loss"])
            log.entropy.append(stats["entropy"])
            log.approx_kl.append(stats["approx_kl"])
            log.clip_frac.append(stats["clip_frac"])
            log.explained_var.append(stats["explained_var"])
            log.lr.append(float(current_lr))
            log.action_freq.append(stats["action_freq"])

            if verbose and (update % log_every == 0 or update == n_updates - 1):
                s, f, l = stats["action_freq"]
                print(
                    f"  upd {update:>4}  ret {ep_rets.mean():>8.2f}  "
                    f"pl {stats['policy_loss']:>7.4f}  vl {stats['value_loss']:>8.3f}  "
                    f"ent {stats['entropy']:.3f}  kl {stats['approx_kl']:.4f}  "
                    f"ev {stats['explained_var']:>6.3f}  "
                    f"S/F/L {s:.2f}/{f:.2f}/{l:.2f}",
                    flush=True,
                )

        return log

    # ---------------------------------------------------------- evaluation

    def evaluate(
        self, env: BinaryMarketEnv, n_episodes: int, deterministic: bool = True
    ) -> dict[str, np.ndarray]:
        """greedy rollout over consecutive episodes. no updates, no exploration."""
        returns, trades, fees, positions = [], [], [], []

        for ep in range(min(n_episodes, len(env.batch))):
            obs, _ = env.reset(options={"episode": ep})
            total, info = 0.0, {}
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, _, _ = self.net.act(obs_t.unsqueeze(0), deterministic=deterministic)
                obs, r, done, _, info = env.step(int(action.item()))
                total += r
                if done:
                    break
            returns.append(total)
            trades.append(info["trades"])
            fees.append(info["fees"])
            positions.append(info["position"])

        return {
            "returns": np.array(returns),
            "trades": np.array(trades),
            "fees": np.array(fees),
            "final_position": np.array(positions),
        }

    def save(self, path: str) -> None:
        torch.save(
            {"model": self.net.state_dict(), "cfg": self.cfg.__dict__}, path
        )

    def load(self, path: str) -> "PPOAgent":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model"])
        return self
