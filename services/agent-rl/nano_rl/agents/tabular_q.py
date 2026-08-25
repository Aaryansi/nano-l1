"""tabular q-learning, used as a sanity check on the environment itself.

this exists to answer one question before any neural network is involved:
**is the mdp learnable at all?** if a tabular learner with nine states cannot
find a signal that is present by construction, the fault is in the environment,
the reward accounting, or the feature plumbing, and no amount of ppo tuning
will fix it.

deliberately minimal: no function approximation, no replay, no target network.
those are exactly the components that can mask an environment bug by learning
something despite it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.synthetic import SIGNAL_IDX, discretize


@dataclass
class TabularQConfig:
    n_bins: int = 9
    n_actions: int = 3
    lr: float = 0.1
    gamma: float = 1.0  # finite-horizon episodes; see docs/MDP.md section 7.1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 500
    seed: int = 0


@dataclass
class TrainLog:
    """per-episode training diagnostics."""

    returns: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)

    def tail_mean(self, k: int = 100) -> float:
        return float(np.mean(self.returns[-k:])) if self.returns else 0.0


class TabularQAgent:
    """epsilon-greedy tabular q-learning over a binned signal feature."""

    def __init__(self, cfg: TabularQConfig | None = None) -> None:
        self.cfg = cfg or TabularQConfig()
        self.q = np.zeros((self.cfg.n_bins + 1, self.cfg.n_actions))
        self._rng = np.random.default_rng(self.cfg.seed)
        self._episode = 0

    def epsilon(self) -> float:
        """linear decay from eps_start to eps_end."""
        frac = min(1.0, self._episode / max(self.cfg.eps_decay_episodes, 1))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, state: int, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon():
            return int(self._rng.integers(0, self.cfg.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> float:
        """one q-learning backup. returns the td error for logging."""
        target = r if done else r + self.cfg.gamma * float(np.max(self.q[s2]))
        td = target - self.q[s, a]
        self.q[s, a] += self.cfg.lr * td
        return float(td)

    def train(
        self,
        env: BinaryMarketEnv,
        n_episodes: int,
        signal_idx: int = SIGNAL_IDX,
    ) -> TrainLog:
        """run q-learning for `n_episodes`, returning the training log.

        note `signal_idx` is an index into the OBSERVATION, not the spot block.
        the caller is responsible for passing the right one; see
        tests/test_agents.py for the mapping.
        """
        log = TrainLog()

        for _ in range(n_episodes):
            obs, _ = env.reset()
            s = discretize(obs, signal_idx, self.cfg.n_bins)
            total, tds = 0.0, []

            while True:
                a = self.act(s)
                obs2, r, done, _, _ = env.step(a)
                s2 = discretize(obs2, signal_idx, self.cfg.n_bins)
                tds.append(abs(self.update(s, a, r, s2, done)))
                total += r
                s = s2
                if done:
                    break

            log.returns.append(total)
            log.epsilons.append(self.epsilon())
            log.td_errors.append(float(np.mean(tds)) if tds else 0.0)
            self._episode += 1

        return log

    def evaluate(
        self,
        env: BinaryMarketEnv,
        n_episodes: int,
        signal_idx: int = SIGNAL_IDX,
    ) -> np.ndarray:
        """greedy rollout returns, one per episode. no exploration, no updates."""
        out = []
        for ep in range(min(n_episodes, len(env.batch))):
            obs, _ = env.reset(options={"episode": ep})
            total = 0.0
            while True:
                s = discretize(obs, signal_idx, self.cfg.n_bins)
                obs, r, done, _, _ = env.step(self.act(s, greedy=True))
                total += r
                if done:
                    break
            out.append(total)
        return np.array(out)
