# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TQC 네트워크: Actor (Gaussian) + Critic (Quantile).

원본 출처:
    drl_agent/scripts/policy/tqc_agent.py (네트워크 부분)
"""

from __future__ import annotations

import torch
import torch.nn as nn

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """가우시안 정책 네트워크 (Tanh squashing).

    Args:
        state_dim:  상태 차원
        action_dim: 행동 차원
        hidden_dim: 은닉층 크기 (기본값: 256)
        activation: 활성화 함수 ("relu" | "elu")
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "relu",
    ):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.ELU()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), act,
            nn.Linear(hidden_dim, hidden_dim), act,
        )
        self.mean_layer    = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterization trick + log-prob (squashed Gaussian).

        Returns:
            action:   tanh-squashed action, shape (batch, action_dim)
            log_prob: log-probability, shape (batch, 1)
        """
        mean, log_std = self(state)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)

        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob  = log_prob.sum(-1, keepdim=True)

        return y_t, log_prob


class QuantileCritic(nn.Module):
    """다중 분위수 크리틱 네트워크 (TQC).

    Args:
        state_dim:  상태 차원
        action_dim: 행동 차원
        n_quantiles: 분위수 개수 (기본값: 25)
        n_critics:   크리틱 네트워크 수 (기본값: 5)
        hidden_dim:  은닉층 크기 (기본값: 256)
        activation:  활성화 함수 ("elu" | "relu")
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 25,
        n_critics: int = 5,
        hidden_dim: int = 256,
        activation: str = "elu",
    ):
        super().__init__()
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), act,
                nn.Linear(hidden_dim, hidden_dim), act,
                nn.Linear(hidden_dim, n_quantiles),
            )
            for _ in range(n_critics)
        ])
        self.n_quantiles = n_quantiles
        self.n_critics   = n_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns shape: (batch, n_critics, n_quantiles)."""
        x = torch.cat([state, action], dim=-1)
        return torch.stack([c(x) for c in self.critics], dim=1)
