# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO 학습 스크립트.

사용 예시:
    # 기본 학습 (4096 환경, 헤드리스)
    ./isaaclab.sh -p scripts/autodrive/train.py \
        --task Isaac-PathTracking-Hunter-v0 \
        --num_envs 4096 --headless

    # 소규모 디버그
    ./isaaclab.sh -p scripts/autodrive/train.py \
        --task Isaac-PathTracking-Hunter-v0 \
        --num_envs 64
"""

import argparse
import sys
import os

# Isaac Lab AppLauncher 설정
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE RSL-RL PPO 학습")
parser.add_argument("--task",       type=str,   default="Isaac-PathTracking-Hunter-v0")
parser.add_argument("--num_envs",   type=int,   default=4096)
parser.add_argument("--seed",       type=int,   default=0)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--log_dir",    type=str,   default="runs")
parser.add_argument("--resume",     type=str,   default=None, help="체크포인트 경로")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab 임포트 (AppLauncher 이후) ────────────────────────────────────
import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.runners import OnPolicyRunner

# 태스크 등록
import isaaclab_autodrive_tasks  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict

# ─────────────────────────────────────────────────────────────────────────────


def main():
    # 환경 생성
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, render_mode=None)

    # 에이전트 설정 로드
    agent_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    module_path, class_name = agent_cfg_entry.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    agent_cfg: RslRlOnPolicyRunnerCfg = getattr(module, class_name)()

    # 설정 오버라이드
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_root = os.path.join(args_cli.log_dir, args_cli.task)
    os.makedirs(log_root, exist_ok=True)

    print_dict(vars(agent_cfg), nesting=0)

    # 학습 실행
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_root, device="cuda:0")

    if args_cli.resume is not None:
        runner.load(args_cli.resume)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
