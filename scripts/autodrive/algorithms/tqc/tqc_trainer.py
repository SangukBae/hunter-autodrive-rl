# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TQC 학습 루프 — Isaac Lab 환경 연동.

병렬 환경(num_envs)에서 수집한 트랜지션을 LAP 리플레이 버퍼에 저장하고
TQC 에이전트를 업데이트합니다.

사용 예시:
    trainer = TQCTrainer(env, cfg, log_dir="logs/tqc/...")
    trainer.train()
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from algorithms.common.buffer import LAP
from algorithms.common.logger import Logger
from algorithms.tqc.tqc_agent import TQCAgent


class TQCTrainer:
    """Isaac Lab 병렬 환경에서 TQC 오프-폴리시 학습.

    Args:
        env:     Isaac Lab gymnasium 환경 (RslRlVecEnvWrapper 또는 DirectRLEnv)
        cfg:     tqc_cfg.yaml 로드 결과 딕셔너리
        log_dir: 체크포인트 / TensorBoard 로그 경로
    """

    def __init__(self, env, cfg: dict[str, Any], log_dir: str):
        self.env     = env
        self.cfg     = cfg
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # ── 디바이스 ──────────────────────────────────────────────────────────
        device_str  = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        # ── 환경 차원 ─────────────────────────────────────────────────────────
        # gymnasium 래퍼(OrderEnforcing 등)는 num_envs를 포워딩하지 않으므로 unwrapped 접근
        self.num_envs   = env.unwrapped.num_envs
        obs_space = env.observation_space
        if hasattr(obs_space, "spaces"):           # Dict 공간 {"policy": Box(...)}
            obs_space = obs_space["policy"]
        self.obs_dim    = obs_space.shape[-1]
        self.action_dim = env.action_space.shape[-1]

        # ── 에이전트 ──────────────────────────────────────────────────────────
        self.agent = TQCAgent(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            actor_lr=cfg.get("actor_lr", 3e-4),
            critic_lr=cfg.get("critic_lr", 3e-4),
            ent_coef_lr=cfg.get("ent_coef_lr", 3e-4),
            n_quantiles=cfg.get("n_quantiles", 25),
            n_critics=cfg.get("n_critics", 5),
            top_quantiles_to_drop_per_net=cfg.get("top_quantiles_to_drop_per_net", 2),
            discount=cfg.get("discount", 0.99),
            tau=cfg.get("tau", 0.005),
            target_entropy=cfg.get("target_entropy", -float(self.action_dim)),
            ent_coef=str(cfg.get("ent_coef", "auto_1.0")),
            actor_hdim=cfg.get("actor_hdim", 256),
            critic_hdim=cfg.get("critic_hdim", 256),
            actor_activ=cfg.get("actor_activ", "relu"),
            critic_activ=cfg.get("critic_activ", "elu"),
        )

        # ── 리플레이 버퍼 ─────────────────────────────────────────────────────
        prioritized = cfg.get("prioritized", False)
        self.buffer = LAP(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            max_size=cfg.get("buffer_size", 1_000_000),
            batch_size=cfg.get("batch_size", 256),
            prioritized=prioritized,
        )

        # ── 학습 스케줄 ───────────────────────────────────────────────────────
        self.total_timesteps  = cfg.get("total_timesteps", 1_000_000)
        self.warmup_steps     = cfg.get("warmup_steps", 25_000)
        self.eval_interval    = cfg.get("eval_interval", 5_000)
        self.eval_episodes    = cfg.get("eval_episodes", 10)
        self.max_episode_steps = cfg.get("max_episode_steps", 2000)

        # ── 로거 ─────────────────────────────────────────────────────────────
        self.logger = Logger(log_dir, use_tensorboard=True)

        # ── 시드 ─────────────────────────────────────────────────────────────
        seed = cfg.get("seed", 0)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # 메인 학습 루프
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """TQC 학습 메인 루프."""
        obs_dict, _ = self.env.reset()
        obs = self._extract_obs(obs_dict)  # (num_envs, obs_dim)

        total_steps = 0           # 총 환경 스텝 (num_envs 개 트랜지션/스텝)
        episode_rewards = np.zeros(self.num_envs)

        print(f"[TQC] 학습 시작: total_timesteps={self.total_timesteps:,}, "
              f"warmup={self.warmup_steps:,}, num_envs={self.num_envs}")

        while total_steps < self.total_timesteps:
            # ── 행동 선택 ──────────────────────────────────────────────────
            if total_steps < self.warmup_steps:
                # 워밍업: 랜덤 행동
                action_np = np.random.uniform(
                    -1.0, 1.0, size=(self.num_envs, self.action_dim)
                ).astype(np.float32)
            else:
                action_np = self.agent.select_action(obs)

            # ── 환경 스텝 ─────────────────────────────────────────────────
            action_tensor = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            next_obs_dict, reward, terminated, truncated, _ = self.env.step(action_tensor)
            next_obs = self._extract_obs(next_obs_dict)  # (num_envs, obs_dim)

            done = (terminated | truncated).cpu().numpy().astype(np.float32)
            rew  = reward.cpu().numpy().reshape(-1)

            # ── 버퍼 저장 (각 env별로 개별 트랜지션 추가) ────────────────────
            for i in range(self.num_envs):
                self.buffer.add(
                    obs[i], action_np[i], next_obs[i], rew[i], done[i]
                )

            obs = next_obs
            episode_rewards += rew
            total_steps += self.num_envs

            # 에피소드 완료된 env 보상 로깅
            for i in range(self.num_envs):
                if done[i] > 0.5:
                    self.logger.log_scalar("train/episode_reward", episode_rewards[i], total_steps)
                    episode_rewards[i] = 0.0

            # ── 정책 업데이트 ─────────────────────────────────────────────
            if total_steps >= self.warmup_steps and len(self.buffer) >= self.buffer.batch_size:
                metrics = self.agent.train(self.buffer)
                if total_steps % 1000 < self.num_envs:
                    for k, v in metrics.items():
                        self.logger.log_scalar(f"train/{k}", v, total_steps)

            # ── 주기적 평가 및 체크포인트 ─────────────────────────────────
            if total_steps % self.eval_interval < self.num_envs:
                eval_reward = self._evaluate()
                self.logger.log_scalar("eval/episode_reward", eval_reward, total_steps)
                print(
                    f"[TQC] steps={total_steps:,} | "
                    f"eval_reward={eval_reward:.3f} | "
                    f"buffer={len(self.buffer):,} | "
                    f"ent_coef={self.agent.ent_coef.item():.4f}"
                )
                ckpt_path = os.path.join(self.log_dir, f"model_{total_steps}.pt")
                self.agent.save(ckpt_path)
                # _evaluate()가 env.reset()을 호출했으므로 obs 갱신
                obs_dict, _ = self.env.reset()
                obs = self._extract_obs(obs_dict)
                episode_rewards[:] = 0.0

        # ── 최종 저장 ─────────────────────────────────────────────────────
        final_path = os.path.join(self.log_dir, "model_final.pt")
        self.agent.save(final_path)
        self.logger.close()
        print(f"[TQC] 학습 완료. 최종 모델: {final_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 평가
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self) -> float:
        """결정론적 정책으로 eval_episodes 에피소드 평균 보상 반환."""
        obs_dict, _ = self.env.reset()
        obs = self._extract_obs(obs_dict)

        total_reward = 0.0
        episode_count = 0
        step_count = 0

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

    # ─────────────────────────────────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_obs(self, obs_dict) -> np.ndarray:
        """isaac lab obs dict → numpy array."""
        if isinstance(obs_dict, dict):
            obs_tensor = obs_dict.get("policy", list(obs_dict.values())[0])
        else:
            obs_tensor = obs_dict
        if isinstance(obs_tensor, torch.Tensor):
            return obs_tensor.cpu().numpy()
        return np.asarray(obs_tensor, dtype=np.float32)
