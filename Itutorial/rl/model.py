"""PPO actor-critic model used by train.py + eval.py."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-feature MLP actor-critic with a state-independent log std.

    Actions come out tanh-squashed to [-1, 1] — the raw Gaussian sample is
    returned alongside so PPO can evaluate the log-prob on the raw sample.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.6))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def act(self, obs: torch.Tensor):
        h = self.features(obs)
        mu = self.mu(h)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(-1)
        squashed = torch.tanh(raw)
        log_prob = log_prob - torch.log(1 - squashed * squashed + 1e-6).sum(-1)
        value = self.v(h).squeeze(-1)
        return squashed, log_prob, value, raw

    def evaluate(self, obs: torch.Tensor, raw: torch.Tensor):
        h = self.features(obs)
        mu = self.mu(h)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(raw).sum(-1)
        squashed = torch.tanh(raw)
        log_prob = log_prob - torch.log(1 - squashed * squashed + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.v(h).squeeze(-1)
        return log_prob, entropy, value
