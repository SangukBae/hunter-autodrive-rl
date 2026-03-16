# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TD7 (TD3 + 7가지 개선) 에이전트.

원본 출처:
    drl_agent/scripts/policy/td7_agent.py
    (Isaac Lab 연동 및 모듈화)

참고 논문:
    "TD7: Re-Establishing Baselines for Offline RL" (Fujimoto & Gu, 2023)
    핵심 구성:
    - SALE (State-Action Learned Embeddings)
    - Checkpointing (성능 후퇴 방지)
    - LAP PER (Latent Action Prioritized Experience Replay)
    - Clipped Double Q-Learning (TD3 스타일)
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 네트워크 정의
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """SALE 상태 인코더.

    상태를 잠재 임베딩으로 변환하여 크리틱에 사용합니다.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(self.net(state), [self.net[-1].out_features])


class TD3Actor(nn.Module):
    """결정론적 정책 네트워크 (TD3 스타일)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TD7Critic(nn.Module):
    """이중 Q-네트워크 크리틱 (잠재 임베딩 사용)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        inp = state_dim + action_dim + latent_dim

        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action, embedding], dim=-1)
        return self.q1(x), self.q2(x)


# ──────────────────────────────────────────────────────────────────────────────
# TD7 에이전트
# ──────────────────────────────────────────────────────────────────────────────

class TD7Agent:
    """TD7 에이전트.

    TD3 기반 + SALE 임베딩 + Checkpointing + LAP PER.

    Args:
        state_dim:   상태 차원
        action_dim:  행동 차원
        device:      연산 디바이스
        **kwargs:    td7_cfg.yaml 하이퍼파라미터
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        # Checkpointing
        reset_weight: float = 0.9,
        steps_before_checkpointing: int = 40_000,
        max_eps_when_checkpointing: int = 50,
        **kwargs,
    ):
        self.device      = device
        self.discount    = discount
        self.tau         = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq

        self.reset_weight               = reset_weight
        self.steps_before_checkpointing = steps_before_checkpointing
        self.max_eps_when_checkpointing = max_eps_when_checkpointing

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder        = Encoder(state_dim, hidden_dim, latent_dim).to(device)
        self.encoder_target = copy.deepcopy(self.encoder)
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor        = TD3Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ── Critic ────────────────────────────────────────────────────────────
        self.critic        = TD7Critic(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()),
            lr=critic_lr,
        )

        # ── Checkpointing ────────────────────────────────────────────────────
        self.checkpoint_actor   = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)
        self.checkpoint_reward  = -float("inf")
        self.eps_since_ckpt     = 0

        self._update_step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 행동 선택
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """상태 → 행동 (numpy 입출력)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            action = self.actor(state_t)
            if not deterministic:
                noise  = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                action = (action + noise).clamp(-1.0, 1.0)
            return action.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # 학습 (미니배치 1회)
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, replay_buffer) -> dict[str, float]:
        """리플레이 버퍼에서 배치 샘플 후 업데이트."""
        self._update_step += 1
        state, action, next_state, reward, not_done = replay_buffer.sample()

        # ── 임베딩 계산 ───────────────────────────────────────────────────────
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            next_emb    = self.encoder_target(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action, next_emb)
            target_q = reward + not_done * self.discount * torch.min(next_q1, next_q2)

        # ── Critic + Encoder 업데이트 ─────────────────────────────────────────
        cur_emb = self.encoder(state)
        cur_q1, cur_q2 = self.critic(state, action, cur_emb)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor 업데이트 (delayed) ──────────────────────────────────────────
        actor_loss_val = 0.0
        if self._update_step % self.policy_freq == 0:
            emb_no_grad = self.encoder(state).detach()
            actor_action = self.actor(state)
            actor_loss = -self.critic.q1(
                torch.cat([state, actor_action, emb_no_grad], dim=-1)
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # ── 소프트 업데이트 ───────────────────────────────────────────────
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for p, tp in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        # ── LAP 우선순위 업데이트 ─────────────────────────────────────────────
        if replay_buffer.prioritized:
            with torch.no_grad():
                td_err = (cur_q1 - target_q).abs()
            replay_buffer.update_priority(td_err.squeeze(-1))

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss_val,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def maybe_update_checkpoint(self, episode_reward: float, total_steps: int) -> bool:
        """에피소드 보상이 개선되면 checkpoint 갱신.

        Returns:
            True이면 체크포인트 업데이트됨
        """
        self.eps_since_ckpt += 1
        if (
            total_steps >= self.steps_before_checkpointing
            and self.eps_since_ckpt >= self.max_eps_when_checkpointing
            and episode_reward > self.checkpoint_reward
        ):
            self.checkpoint_reward  = episode_reward
            self.checkpoint_actor   = copy.deepcopy(self.actor)
            self.checkpoint_encoder = copy.deepcopy(self.encoder)
            self.eps_since_ckpt     = 0
            return True
        return False

    def restore_from_checkpoint(self) -> None:
        """현재 actor를 checkpoint와 가중 평균."""
        for p, cp in zip(self.actor.parameters(), self.checkpoint_actor.parameters()):
            p.data.copy_(self.reset_weight * cp.data + (1.0 - self.reset_weight) * p.data)
        for p, cp in zip(self.encoder.parameters(), self.checkpoint_encoder.parameters()):
            p.data.copy_(self.reset_weight * cp.data + (1.0 - self.reset_weight) * p.data)

    # ─────────────────────────────────────────────────────────────────────────
    # 체크포인트 파일
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor":   self.actor.state_dict(),
                "critic":  self.critic.state_dict(),
                "encoder": self.encoder.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor_target   = copy.deepcopy(self.actor)
        self.critic_target  = copy.deepcopy(self.critic)
        self.encoder_target = copy.deepcopy(self.encoder)
