# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""학습된 PPO 정책 시각화 스크립트.

사용 예시:
    ./isaaclab.sh -p scripts/autodrive/play.py \
        --task Isaac-PathTracking-Hunter-Play-v0 \
        --checkpoint runs/Isaac-PathTracking-Hunter-v0/model_1000.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE 정책 시각화")
parser.add_argument("--task",       type=str,  default="Isaac-PathTracking-Hunter-Play-v0")
parser.add_argument("--num_envs",   type=int,  default=16)
parser.add_argument("--checkpoint", type=str,  required=True, help="모델 체크포인트 경로 (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # 시각화는 항상 렌더링 ON

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl.runners import OnPolicyRunner
import isaaclab_autodrive_tasks  # noqa: F401


def main():
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, render_mode="rgb_array")

    agent_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    module_path, class_name = agent_cfg_entry.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    agent_cfg = getattr(module, class_name)()

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)

    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.no_grad():
            action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
