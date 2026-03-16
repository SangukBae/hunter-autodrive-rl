# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TQC (Truncated Quantile Critics) 에이전트.

원본 출처:
    drl_agent/scripts/policy/tqc_agent.py
    (Isaac Lab 연동 및 LAP PER 통합)

참고 논문:
    "Controlling Overestimation Bias with Truncated Mixture of Continuous
     Distributional Quantile Critics" (Kuznetsov et al., 2020)
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, QuantileCritic


class TQCAgent:
    """TQC 에이전트.

    Actor (Gaussian) + n_critics 개의 Quantile Critic + 자동 엔트로피 조정.
    상위 분위수 제거(top-quantile dropping)로 과대추정 억제.

    Args:
        state_dim:    상태 차원
        action_dim:   행동 차원
        device:       연산 디바이스
        **kwargs:     tqc_cfg.yaml 항목과 동일한 하이퍼파라미터
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        ent_coef_lr: float = 3e-4,
        n_quantiles: int = 25,
        n_critics: int = 5,
        top_quantiles_to_drop_per_net: int = 2,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: float = -2.0,
        ent_coef: str = "auto_1.0",
        actor_hdim: int = 256,
        critic_hdim: int = 256,
        actor_activ: str = "relu",
        critic_activ: str = "elu",
        **kwargs,
    ):
        self.device   = device
        self.discount = discount
        self.tau      = tau
        self.n_quantiles = n_quantiles
        self.n_critics   = n_critics
        self.top_quantiles_to_drop = top_quantiles_to_drop_per_net * n_critics
        self.target_entropy = target_entropy

        # ── Actor ────────────────────────────────────────────────────────────
        self.actor = Actor(state_dim, action_dim, actor_hdim, actor_activ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ── Critic ───────────────────────────────────────────────────────────
        self.critic = QuantileCritic(
            state_dim, action_dim, n_quantiles, n_critics, critic_hdim, critic_activ
        ).to(device)
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

        # target quantile 수 (전체 - top 제거)
        self.n_target_quantiles = n_quantiles * n_critics - self.top_quantiles_to_drop

    @property
    def ent_coef(self) -> torch.Tensor:
        return self.log_ent_coef.exp()

    # ─────────────────────────────────────────────────────────────────────────
    # 행동 선택
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """상태 배열로부터 행동 선택 (numpy 입출력).

        Args:
            state:        shape (obs_dim,) 또는 (num_envs, obs_dim)
            deterministic: True이면 mean 행동 반환 (평가/배포용)
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            if deterministic:
                mean, _ = self.actor(state_t)
                action  = torch.tanh(mean)
            else:
                action, _ = self.actor.get_action(state_t)
            return action.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # 학습 (미니배치 1회 업데이트)
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, replay_buffer) -> dict[str, float]:
        """리플레이 버퍼에서 배치 샘플 후 actor/critic/entropy 업데이트.

        Returns:
            딕셔너리 {"critic_loss", "actor_loss", "ent_coef"}
        """
        state, action, next_state, reward, not_done = replay_buffer.sample()

        # ── Critic 타깃 계산 ─────────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_pi = self.actor.get_action(next_state)
            # (batch, n_critics, n_quantiles) → (batch, n_critics * n_quantiles)
            next_z = self.critic_target(next_state, next_action)
            next_z = next_z.view(next_z.shape[0], -1)
            # 상위 분위수 제거 (정렬 후 하위만 사용)
            next_z, _ = torch.sort(next_z, dim=1)
            next_z = next_z[:, : self.n_target_quantiles]

            # 엔트로피 보정 + 벨만 타깃
            target_z = reward + not_done * self.discount * (
                next_z - self.ent_coef * next_log_pi
            )  # (batch, n_target_quantiles)

        # ── Critic 업데이트 ───────────────────────────────────────────────────
        cur_z = self.critic(state, action)  # (batch, n_critics, n_quantiles)
        # huber quantile loss
        cur_z_flat  = cur_z.view(cur_z.shape[0], -1, 1)           # (B, C*Q, 1)
        target_z_exp = target_z.unsqueeze(1)                       # (B, 1, T)
        td_err = target_z_exp - cur_z_flat                         # (B, C*Q, T)

        huber = torch.where(td_err.abs() <= 1.0, 0.5 * td_err ** 2, td_err.abs() - 0.5)
        n_cur = cur_z_flat.shape[1]
        tau   = (torch.arange(n_cur, device=self.device, dtype=torch.float32) + 0.5) / n_cur
        tau   = tau.unsqueeze(0).unsqueeze(-1)                    # (1, C*Q, 1)
        critic_loss = (torch.abs(tau - (td_err < 0).float()) * huber).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor 업데이트 ────────────────────────────────────────────────────
        new_action, log_pi = self.actor.get_action(state)
        new_z = self.critic(state, new_action)                    # (B, n_critics, n_quantiles)
        actor_loss = (self.ent_coef * log_pi - new_z.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)).mean()

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

        # ── LAP 우선순위 업데이트 ─────────────────────────────────────────────
        if replay_buffer.prioritized:
            with torch.no_grad():
                priority = td_err.detach().abs().mean(dim=(1, 2)).sqrt()
            replay_buffer.update_priority(priority)

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
            {
                "actor":  self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
