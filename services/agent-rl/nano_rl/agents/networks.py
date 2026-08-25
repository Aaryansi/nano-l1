"""actor-critic network for ppo.

a small mlp with a shared trunk and two heads. the value head matters as much
as the policy head here: docs/MDP.md section 1.2 makes the critic's V(s)
checkable against realised settlement frequency, which is the phase-5c
"explain the value predictions" target. so the critic is a deliverable, not
just a variance-reduction device.

sizing: 18 inputs, two hidden layers of 64. deliberately small. the corpus has
~90k transitions and no exploitable signal, so a larger network would only
memorise noise faster.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """orthogonal weight init, the ppo convention.

    the small gain on the policy output layer matters: it starts the policy
    close to uniform, so early training does not commit to an action before
    the critic has any idea what states are worth.
    """
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """shared-trunk categorical actor with a scalar critic."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            orthogonal_init(nn.Linear(n_features, hidden)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        # gain 0.01 keeps the initial policy near-uniform
        self.policy_head = orthogonal_init(nn.Linear(hidden, n_actions), gain=0.01)
        # gain 1.0 for a value head predicting unbounded returns
        self.value_head = orthogonal_init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """returns (action logits, state value)."""
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample an action. returns (action, log_prob, value).

        `deterministic` takes the argmax instead, which is what evaluation
        uses so that reported test performance is not a lucky sample.
        """
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """log-probs, entropy, and values for stored transitions.

        used during the update, where the policy has moved since the actions
        were sampled. this is what makes the importance ratio meaningful.
        """
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

    @torch.no_grad()
    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """policy distribution, for explainability and diagnostics."""
        logits, _ = self(obs)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """critic output alone, for the calibration analysis."""
        return self(obs)[1]
