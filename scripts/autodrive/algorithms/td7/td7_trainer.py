# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TD7 학습 루프 — Isaac Lab 환경 연동."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from algorithms.common.buffer import LAP
from algorithms.common.logger import Logger
from algorithms.td7.td7_agent import TD7Agent


class TD7Trainer:
    """Isaac Lab 병렬 환경에서 TD7 학습.

    Args:
        env:     Isaac Lab gymnasium 환경
        cfg:     td7_cfg.yaml 로드 결과 딕셔너리
        log_dir: 체크포인트 / TensorBoard 로그 경로
    """

    def __init__(self, env, cfg: dict[str, Any], log_dir: str):
        self.env     = env
        self.cfg     = cfg
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        device_str  = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        # gymnasium 래퍼(OrderEnforcing 등)는 num_envs를 포워딩하지 않으므로 unwrapped 접근
        self.num_envs   = env.unwrapped.num_envs
        obs_space = env.observation_space
        if hasattr(obs_space, "spaces"):           # Dict 공간 {"policy": Box(...)}
            obs_space = obs_space["policy"]
        self.obs_dim    = obs_space.shape[-1]
        self.action_dim = env.action_space.shape[-1]

        # ── 에이전트 ──────────────────────────────────────────────────────────
        self.agent = TD7Agent(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            actor_lr=cfg.get("actor_lr", 3e-4),
            critic_lr=cfg.get("critic_lr", 3e-4),
            discount=cfg.get("discount", 0.99),
            tau=cfg.get("tau", 0.005),
            policy_noise=cfg.get("policy_noise", 0.2),
            noise_clip=cfg.get("noise_clip", 0.5),
            policy_freq=cfg.get("policy_freq", 2),
            hidden_dim=cfg.get("hidden_dim", 256),
            latent_dim=cfg.get("latent_dim", 256),
            reset_weight=cfg.get("reset_weight", 0.9),
            steps_before_checkpointing=cfg.get("steps_before_checkpointing", 40_000),
            max_eps_when_checkpointing=cfg.get("max_eps_when_checkpointing", 50),
        )

        # ── 리플레이 버퍼 ─────────────────────────────────────────────────────
        prioritized = cfg.get("prioritized", True)  # TD7는 기본 LAP PER 사용
        self.buffer = LAP(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            max_size=cfg.get("buffer_size", 1_000_000),
            batch_size=cfg.get("batch_size", 256),
            prioritized=prioritized,
        )

        # ── 학습 스케줄 ───────────────────────────────────────────────────────
        self.total_timesteps   = cfg.get("total_timesteps", 1_000_000)
        self.warmup_steps      = cfg.get("warmup_steps", 25_000)
        self.eval_interval     = cfg.get("eval_interval", 5_000)
        self.eval_episodes     = cfg.get("eval_episodes", 10)
        self.max_episode_steps = cfg.get("max_episode_steps", 2000)

        self.logger = Logger(log_dir, use_tensorboard=True)

        seed = cfg.get("seed", 0)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # 메인 학습 루프
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """TD7 학습 메인 루프."""
        obs_dict, _ = self.env.reset()
        obs = self._extract_obs(obs_dict)

        total_steps    = 0
        episode_rewards = np.zeros(self.num_envs)
        best_eval_reward = -float("inf")

        print(f"[TD7] 학습 시작: total_timesteps={self.total_timesteps:,}, "
              f"warmup={self.warmup_steps:,}, num_envs={self.num_envs}")

        while total_steps < self.total_timesteps:
            # ── 행동 선택 ──────────────────────────────────────────────────
            if total_steps < self.warmup_steps:
                action_np = np.random.uniform(
                    -1.0, 1.0, size=(self.num_envs, self.action_dim)
                ).astype(np.float32)
            else:
                action_np = self.agent.select_action(obs, deterministic=False)

            # ── 환경 스텝 ─────────────────────────────────────────────────
            action_tensor = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            next_obs_dict, reward, terminated, truncated, _ = self.env.step(action_tensor)
            next_obs = self._extract_obs(next_obs_dict)

            done = (terminated | truncated).cpu().numpy().astype(np.float32)
            rew  = reward.cpu().numpy().reshape(-1)

            for i in range(self.num_envs):
                self.buffer.add(obs[i], action_np[i], next_obs[i], rew[i], done[i])

            obs = next_obs
            episode_rewards += rew
            total_steps += self.num_envs

            # ── 에피소드 완료: checkpointing ─────────────────────────────
            for i in range(self.num_envs):
                if done[i] > 0.5:
                    self.logger.log_scalar("train/episode_reward", episode_rewards[i], total_steps)
                    updated = self.agent.maybe_update_checkpoint(episode_rewards[i], total_steps)
                    if updated:
                        self.logger.log_scalar(
                            "train/checkpoint_reward", episode_rewards[i], total_steps
                        )
                    episode_rewards[i] = 0.0

            # ── 정책 업데이트 ─────────────────────────────────────────────
            if total_steps >= self.warmup_steps and len(self.buffer) >= self.buffer.batch_size:
                # LAP: 버퍼 max_priority 주기적 초기화
                if self.buffer.prioritized and total_steps % 2500 < self.num_envs:
                    self.buffer.reset_max_priority()

                metrics = self.agent.train(self.buffer)
                if total_steps % 1000 < self.num_envs:
                    for k, v in metrics.items():
                        self.logger.log_scalar(f"train/{k}", v, total_steps)

            # ── 주기적 평가 ───────────────────────────────────────────────
            if total_steps % self.eval_interval < self.num_envs:
                eval_reward = self._evaluate()
                self.logger.log_scalar("eval/episode_reward", eval_reward, total_steps)
                print(
                    f"[TD7] steps={total_steps:,} | "
                    f"eval_reward={eval_reward:.3f} | "
                    f"buffer={len(self.buffer):,}"
                )
                ckpt_path = os.path.join(self.log_dir, f"model_{total_steps}.pt")
                self.agent.save(ckpt_path)

                # checkpoint 이하로 성능 후퇴 시 복구
                if (
                    total_steps >= self.agent.steps_before_checkpointing
                    and eval_reward < self.agent.checkpoint_reward * 0.8
                ):
                    print("[TD7] 성능 후퇴 감지 → checkpoint 복원")
                    self.agent.restore_from_checkpoint()
                # _evaluate()가 env.reset()을 호출했으므로 obs 갱신
                obs_dict, _ = self.env.reset()
                obs = self._extract_obs(obs_dict)
                episode_rewards[:] = 0.0

        final_path = os.path.join(self.log_dir, "model_final.pt")
        self.agent.save(final_path)
        self.logger.close()
        print(f"[TD7] 학습 완료. 최종 모델: {final_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 평가
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self) -> float:
        obs_dict, _ = self.env.reset()
        obs = self._extract_obs(obs_dict)

        total_reward  = 0.0
        episode_count = 0
        step_count    = 0

        while episode_count < self.eval_episodes:
            action_np = self.agent.select_action(obs, deterministic=True)
            action_tensor = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            next_obs_dict, reward, terminated, truncated, _ = self.env.step(action_tensor)
            next_obs = self._extract_obs(next_obs_dict)

            done = (terminated | truncated).cpu().numpy()
            rew  = reward.cpu().numpy().reshape(-1)

            total_reward  += rew[0]
            step_count    += 1
            episode_count += int(done[0])

            if step_count >= self.max_episode_steps:
                break

            obs = next_obs

        return total_reward / max(episode_count, 1)

    def _extract_obs(self, obs_dict) -> np.ndarray:
        if isinstance(obs_dict, dict):
            obs_tensor = obs_dict.get("policy", list(obs_dict.values())[0])
        else:
            obs_tensor = obs_dict
        if isinstance(obs_tensor, torch.Tensor):
            return obs_tensor.cpu().numpy()
        return np.asarray(obs_tensor, dtype=np.float32)
