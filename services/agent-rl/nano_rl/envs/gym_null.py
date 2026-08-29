"""the null-model test on standard gymnasium environments.

everything else in this project is measured on one financial market, which
makes the results a case study rather than a method. this module ports the test
to CartPole so the claim can be checked somewhere with no relationship to
trading, no transaction costs, and a well understood optimal policy.

two things get better in the move, not worse:

  exact shapley. CartPole has four observation features, so all 2^4 = 16
  coalitions can be enumerated. there is no sampling error at all, which
  removes the one place a sceptic could locate the effect.

  a competence axis. an agent can be checkpointed through training, giving a
  sequence of policies from useless to converged on an identical task. that
  turns "does the test detect a competent agent" into a curve rather than a
  single comparison.

the null construction is the part the Atari saliency literature called
non-obvious for deep RL, since there is no dataset to permute. the answer used
here is to corrupt the OBSERVATION CHANNEL while leaving the dynamics and reward
untouched: the agent trains normally, collects real reward, and simply cannot
condition on anything. that is a structure-free counterpart of an arbitrary
environment, and it needs no access to the task's internals.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from nano_rl.agents.networks import ActorCritic


class BlindObservation(gym.ObservationWrapper):
    """replace every observation with a draw from a fixed reference distribution.

    the environment's dynamics and reward are untouched, so the agent still
    faces the real task and still receives real return. it simply has no usable
    observation. an agent trained here is the reinforcement-learning analogue
    of a supervised model trained on permuted labels: normally trained, on a
    real objective, with nothing to learn from its input.

    the reference moments are estimated once from a random rollout so that the
    fake observations occupy the same range as real ones. that matters because
    a network fed out-of-range inputs would fail for reasons of scale rather
    than of information, which would be a different experiment.
    """

    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray, seed: int = 0):
        super().__init__(env)
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def observation(self, observation):  # noqa: D102
        return self._rng.normal(self._mean, self._std).astype(np.float32)


def observation_moments(env_id: str, n_steps: int = 4000, seed: int = 0):
    """mean and std of the observation under a random policy."""
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    for _ in range(n_steps):
        rows.append(obs)
        obs, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    arr = np.asarray(rows, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-8


@dataclass
class GymPPOConfig:
    """a smaller ppo than the trading one; CartPole needs far less."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    steps_per_batch: int = 2048
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden: int = 64
    seed: int = 0


def train_gym_ppo(
    env: gym.Env,
    cfg: GymPPOConfig,
    total_steps: int,
    checkpoint_fractions: tuple[float, ...] = (),
) -> tuple[ActorCritic, list[tuple[float, ActorCritic]]]:
    """ppo on a gym env, returning the final net and any requested checkpoints.

    written separately from nano_rl/agents/ppo.py rather than generalising it.
    the trading agent has episode-structured rollouts, gamma=1, and a terminal
    settlement, none of which apply here, and bending one implementation to
    cover both would put conditionals through the part of the code the whole
    project's correctness rests on.

    truncation is handled properly: a time-limit cutoff bootstraps from the
    value function, while a genuine termination does not. conflating them is
    the standard CartPole bug and it makes the agent believe falling over and
    surviving to the limit are the same event.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    net = ActorCritic(obs_dim, n_actions, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    checkpoints: list[tuple[float, ActorCritic]] = []
    wanted = sorted(checkpoint_fractions)

    obs, _ = env.reset(seed=cfg.seed)
    steps_done = 0

    while steps_done < total_steps:
        ob_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []
        done_buf, trunc_val_buf = [], []

        for _ in range(cfg.steps_per_batch):
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, logp, v = net.act(t)
            nobs, r, term, trunc, _ = env.step(int(a.item()))

            ob_buf.append(obs)
            act_buf.append(int(a.item()))
            logp_buf.append(float(logp.item()))
            val_buf.append(float(v.item()))
            rew_buf.append(float(r))
            done_buf.append(bool(term or trunc))

            # on a time-limit truncation the future is not worthless, so the
            # bootstrap value of the next state is recorded and used below.
            if trunc and not term:
                with torch.no_grad():
                    nv = net.value(
                        torch.as_tensor(nobs, dtype=torch.float32).unsqueeze(0)
                    )
                trunc_val_buf.append(float(nv.item()))
            else:
                trunc_val_buf.append(0.0)

            obs = nobs
            if term or trunc:
                obs, _ = env.reset()

        steps_done += cfg.steps_per_batch

        rewards = np.asarray(rew_buf)
        values = np.asarray(val_buf)
        dones = np.asarray(done_buf)
        boots = np.asarray(trunc_val_buf)

        with torch.no_grad():
            last_v = float(
                net.value(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
            )

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # boots[t] is 0.0 on a true termination and the bootstrapped
                # value of the next state on a time-limit truncation. either
                # way the episode boundary stops gae propagating backwards
                # across it, so next_nonterminal is 0 in both cases.
                next_v = boots[t]
                next_nonterminal = 0.0
            else:
                next_v = values[t + 1] if t + 1 < len(rewards) else last_v
                next_nonterminal = 1.0
            delta = rewards[t] + cfg.gamma * next_v - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * gae
            adv[t] = gae
        returns = adv + values

        ob_t = torch.as_tensor(np.asarray(ob_buf), dtype=torch.float32)
        act_t = torch.as_tensor(act_buf, dtype=torch.long)
        old_logp = torch.as_tensor(logp_buf, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        idx = np.arange(len(ob_buf))
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), cfg.minibatch_size):
                mb = torch.as_tensor(
                    idx[start : start + cfg.minibatch_size], dtype=torch.long
                )
                if len(mb) < 2:
                    continue
                lp, ent, v = net.evaluate_actions(ob_t[mb], act_t[mb])
                ratio = (lp - old_logp[mb]).exp()
                a_mb = adv_t[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)

                pg = torch.max(
                    -a_mb * ratio,
                    -a_mb * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()
                vl = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                loss = pg + cfg.value_coef * vl - cfg.entropy_coef * ent.mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        frac = steps_done / total_steps
        while wanted and frac >= wanted[0]:
            snap = ActorCritic(obs_dim, n_actions, cfg.hidden)
            snap.load_state_dict(net.state_dict())
            checkpoints.append((wanted.pop(0), snap))

    return net, checkpoints


def evaluate_gym(net: ActorCritic, env_id: str, n_episodes: int = 30, seed: int = 0) -> float:
    """mean undiscounted return under the greedy policy."""
    env = gym.make(env_id)
    total = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        ep = 0.0
        while True:
            with torch.no_grad():
                a, _, _ = net.act(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                    deterministic=True,
                )
            obs, r, term, trunc, _ = env.step(int(a.item()))
            ep += float(r)
            if term or trunc:
                break
        total.append(ep)
    env.close()
    return float(np.mean(total))


def masked_return(
    net: ActorCritic,
    env_id: str,
    mask: np.ndarray,
    background: np.ndarray,
    n_episodes: int = 20,
    seed: int = 0,
) -> float:
    """mean return when the agent observes only the features in `mask`.

    features outside the mask are replaced by a background draw at every step,
    resampled each step rather than fixed per episode, so that consistency of
    the fake values cannot itself carry information.
    """
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    total = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        ep = 0.0
        while True:
            x = np.asarray(obs, dtype=np.float32).copy()
            if not mask.all():
                draw = background[rng.integers(0, len(background))]
                x[~mask] = draw[~mask]
            with torch.no_grad():
                a, _, _ = net.act(
                    torch.as_tensor(x, dtype=torch.float32).unsqueeze(0),
                    deterministic=True,
                )
            obs, r, term, trunc, _ = env.step(int(a.item()))
            ep += float(r)
            if term or trunc:
                break
        total.append(ep)

    env.close()
    return float(np.mean(total))


def attribution_span_fast(
    net: ActorCritic,
    env_id: str,
    background: np.ndarray,
    n_features: int,
    n_episodes: int = 20,
    seed: int = 0,
) -> float:
    """v(all) - v(none) in TWO rollouts rather than 2^n.

    the efficiency axiom gives sum_i phi_i = v(N) - v(empty), so the span, which
    is the null test's entire statistic, needs only the two endpoint
    evaluations. enumerating every coalition to recover it is exact but
    wasteful: 8x the work on CartPole and 32x on Acrobot for a number already
    available from the endpoints.

    the per-feature values still require the full enumeration, so
    exact_shapley_span remains for when those are wanted.
    """
    full = masked_return(
        net, env_id, np.ones(n_features, dtype=bool), background,
        n_episodes=n_episodes, seed=seed,
    )
    empty = masked_return(
        net, env_id, np.zeros(n_features, dtype=bool), background,
        n_episodes=n_episodes, seed=seed,
    )
    return float(full - empty)


def exact_shapley_span(
    net: ActorCritic,
    env_id: str,
    background: np.ndarray,
    n_features: int,
    n_episodes: int = 20,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """exact shapley values by enumerating all 2^n coalitions.

    feasible because CartPole has four features. returns (span, values), where
    the span is v(all) - v(none) and equals the sum of the values exactly, by
    the efficiency axiom. no sampling, so any effect cannot be blamed on
    estimator noise.
    """
    cache: dict[tuple[bool, ...], float] = {}

    def v(mask: np.ndarray) -> float:
        key = tuple(bool(b) for b in mask)
        if key not in cache:
            cache[key] = masked_return(
                net, env_id, mask, background, n_episodes=n_episodes, seed=seed
            )
        return cache[key]

    from math import factorial

    n = n_features
    values = np.zeros(n)
    others = list(range(n))

    for i in range(n):
        rest = [j for j in others if j != i]
        for size in range(len(rest) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in itertools.combinations(rest, size):
                m = np.zeros(n, dtype=bool)
                m[list(combo)] = True
                without = v(m)
                m[i] = True
                with_i = v(m)
                values[i] += weight * (with_i - without)

    span = v(np.ones(n, dtype=bool)) - v(np.zeros(n, dtype=bool))
    return float(span), values
