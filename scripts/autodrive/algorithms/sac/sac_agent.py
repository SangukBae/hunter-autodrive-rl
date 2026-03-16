# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""SAC (Soft Actor-Critic) 에이전트.

참고 논문:
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
     Learning with a Stochastic Actor" (Haarnoja et al., 2018)

TQC와의 차이:
    - Quantile Critic 대신 일반 MLP Critic (Double Q)
    - 상위 분위수 제거 없음
    - 구조가 단순하여 빠른 학습 가능
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACActorNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x       = self.net(state)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob  = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob


class SACCriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        inp = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    """SAC 에이전트 (Double Q-Critic + 자동 엔트로피 조정).

    Args:
        state_dim:      상태 차원
        action_dim:     행동 차원
        device:         연산 디바이스
        actor_lr:       Actor 학습률
        critic_lr:      Critic 학습률
        ent_coef_lr:    엔트로피 계수 학습률
        discount:       할인율
        tau:            Polyak 평균 계수
        target_entropy: 목표 엔트로피 (기본값: -action_dim)
        ent_coef:       초기 엔트로피 계수 ("auto_<val>" 또는 고정값)
        hidden_dim:     은닉층 크기
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        ent_coef_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        ent_coef: str = "auto_1.0",
        hidden_dim: int = 256,
        **kwargs,
    ):
        self.device   = device
        self.discount = discount
        self.tau      = tau
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor = SACActorNet(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ── Critic ────────────────────────────────────────────────────────────
        self.critic = SACCriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── 엔트로피 계수 ─────────────────────────────────────────────────────
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            init_val = float(ent_coef.split("_")[1]) if "_" in ent_coef else 1.0
            self.log_ent_coef = torch.tensor(
                np.log(init_val), dtype=torch.float32, device=device, requires_grad=True
            )
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=ent_coef_lr)
            self.auto_entropy = True
        else:
            self.log_ent_coef = torch.tensor(
                np.log(float(ent_coef)), dtype=torch.float32, device=device
            )
            self.auto_entropy = False

    @property
    def ent_coef(self) -> torch.Tensor:
        return self.log_ent_coef.exp()

    # ─────────────────────────────────────────────────────────────────────────
    # 행동 선택
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            if deterministic:
                mean, _ = self.actor(state_t)
                return torch.tanh(mean).cpu().numpy()
            action, _ = self.actor.get_action(state_t)
            return action.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # 학습
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, replay_buffer) -> dict[str, float]:
        state, action, next_state, reward, not_done = replay_buffer.sample()

        # ── Critic 타깃 ───────────────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_pi = self.actor.get_action(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next  = torch.min(q1_next, q2_next) - self.ent_coef * next_log_pi
            target_q = reward + not_done * self.discount * q_next

        # ── Critic 업데이트 ───────────────────────────────────────────────────
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor 업데이트 ────────────────────────────────────────────────────
        new_action, log_pi = self.actor.get_action(state)
        q1_new, q2_new = self.critic(state, new_action)
        actor_loss = (self.ent_coef * log_pi - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── 엔트로피 계수 업데이트 ────────────────────────────────────────────
        if self.auto_entropy:
            ent_loss = -(self.log_ent_coef * (log_pi + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_loss.backward()
            self.ent_coef_optimizer.step()

        # ── Target network 소프트 업데이트 ───────────────────────────────────
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "ent_coef":    self.ent_coef.item(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 체크포인트
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
