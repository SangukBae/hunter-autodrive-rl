# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""학습된 TQC / TD7 정책 시각화 및 평가 스크립트.

사용 예시:
    # TQC 정책 시각화 (GUI)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
        --task Isaac-LidarNav-Hunter-Play-v0 \
        --algo tqc \
        --checkpoint logs/tqc/hunter_tqc/2024-01-01_00-00-00/model_final.pt \
        --num_envs 4

    # TD7 정책 시각화
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
        --task Isaac-LidarNav-Hunter-Play-v0 \
        --algo td7 \
        --checkpoint logs/td7/hunter_td7/2024-01-01_00-00-00/model_final.pt

    # PhaseE 환경에서 평가
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
        --task Isaac-LidarNav-Hunter-PhaseE-v0 \
        --algo tqc \
        --checkpoint path/to/model.pt \
        --eval_episodes 100 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LiDAR Navigation 정책 시각화/평가")
parser.add_argument("--task",           type=str,  default="Isaac-LidarNav-Hunter-Play-v0")
parser.add_argument("--algo",           type=str,  default="tqc", choices=["tqc", "td7"])
parser.add_argument("--checkpoint",     type=str,  required=True,  help="모델 체크포인트 경로 (.pt)")
parser.add_argument("--num_envs",       type=int,  default=4)
parser.add_argument("--eval_episodes",  type=int,  default=20,     help="평가 에피소드 수")
parser.add_argument("--max_steps",      type=int,  default=3000,   help="에피소드당 최대 스텝")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import numpy as np
import torch
import gymnasium as gym

import isaaclab_autodrive_tasks  # noqa: F401 — 환경 등록

sys.path.insert(0, os.path.dirname(__file__))
from algorithms.tqc import TQCAgent
from algorithms.td7 import TD7Agent


def _extract_obs(obs_dict) -> np.ndarray:
    if isinstance(obs_dict, dict):
        obs_tensor = obs_dict.get("policy", list(obs_dict.values())[0])
    else:
        obs_tensor = obs_dict
    if isinstance(obs_tensor, torch.Tensor):
        return obs_tensor.cpu().numpy()
    return np.asarray(obs_tensor, dtype=np.float32)


def main():
    device_str = args_cli.device if args_cli.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)

    # ── 환경 생성 ─────────────────────────────────────────────────────────────
    import importlib
    spec      = gym.spec(args_cli.task)
    cfg_entry = spec.kwargs["env_cfg_entry_point"]
    mod_path, cls_name = cfg_entry.rsplit(":", 1)
    cfg_mod   = importlib.import_module(mod_path)
    env_cfg   = getattr(cfg_mod, cls_name)()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)

    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        obs_space = obs_space["policy"]
    obs_dim    = obs_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    # ── 에이전트 로드 ─────────────────────────────────────────────────────────
    if args_cli.algo == "tqc":
        agent = TQCAgent(state_dim=obs_dim, action_dim=action_dim, device=device)
    else:
        agent = TD7Agent(state_dim=obs_dim, action_dim=action_dim, device=device)

    agent.load(args_cli.checkpoint)
    print(f"[PLAY] 체크포인트 로드: {args_cli.checkpoint}")
    print(f"[PLAY] 알고리즘: {args_cli.algo.upper()}, 태스크: {args_cli.task}, "
          f"num_envs: {args_cli.num_envs}")

    # ── 평가 루프 ─────────────────────────────────────────────────────────────
    obs_dict, _ = env.reset()
    obs = _extract_obs(obs_dict)

    episode_rewards  = np.zeros(args_cli.num_envs)
    episode_lengths  = np.zeros(args_cli.num_envs, dtype=int)
    completed_eps    = []
    goal_reached_cnt = 0
    collision_cnt    = 0
    total_eps_needed = args_cli.eval_episodes
    total_step       = 0

    while len(completed_eps) < total_eps_needed:
        action_np = agent.select_action(obs, deterministic=True)
        action_t  = torch.tensor(action_np, dtype=torch.float32, device=device)

        next_obs_dict, reward, terminated, truncated, info = env.step(action_t)
        next_obs = _extract_obs(next_obs_dict)

        done  = (terminated | truncated).cpu().numpy()
        rew   = reward.cpu().numpy().reshape(-1)
        term  = terminated.cpu().numpy()

        episode_rewards += rew
        episode_lengths += 1
        total_step      += args_cli.num_envs

        for i in range(args_cli.num_envs):
            if done[i]:
                completed_eps.append(episode_rewards[i])
                if term[i]:   # terminated (not timeout) → goal or collision
                    # heuristic: large positive reward at end → goal reached
                    if episode_rewards[i] > 50.0:
                        goal_reached_cnt += 1
                    else:
                        collision_cnt += 1
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

        if total_step >= args_cli.max_steps * total_eps_needed:
            break

        obs = next_obs

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    n = len(completed_eps)
    if n > 0:
        mean_rew = np.mean(completed_eps)
        std_rew  = np.std(completed_eps)
        print(f"\n{'='*55}")
        print(f"[PLAY] 평가 완료: {n}개 에피소드")
        print(f"  평균 보상  : {mean_rew:.2f} ± {std_rew:.2f}")
        print(f"  목표 도달  : {goal_reached_cnt} / {n} "
              f"({100*goal_reached_cnt/n:.1f}%)")
        print(f"  충돌 종료  : {collision_cnt} / {n} "
              f"({100*collision_cnt/n:.1f}%)")
        print(f"{'='*55}")
    else:
        print("[PLAY] 완료된 에피소드 없음.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
