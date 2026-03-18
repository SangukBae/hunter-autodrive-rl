# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-LidarNav-Hunter-v0 빠른 동작 검증 스크립트.

사용:
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/test_lidar_nav.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LidarNav env 동작 검증")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import torch
import gymnasium as gym

import isaaclab_autodrive_tasks  # noqa: F401 — 환경 등록


def main():
    import importlib

    task = "Isaac-LidarNav-Hunter-v0"
    spec = gym.spec(task)
    cfg_entry = spec.kwargs["env_cfg_entry_point"]   # "module:ClassName"
    mod_path, cls_name = cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(mod_path), cls_name)()

    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()

    policy_obs = obs["policy"]
    print(f"\n{'='*50}")
    print(f"obs shape  : {policy_obs.shape}")          # 기대: (64, 82)
    print(f"obs sample : {policy_obs[0, :5]}")         # 기대: [0,1] 범위 float
    assert policy_obs.shape[-1] == 82, f"관측 차원 오류: {policy_obs.shape[-1]} != 82"
    # LiDAR [0:80]: [0,1] / goal_dist [80]: [0, ~3] (대각선 거리 / map_size) / goal_angle [81]: [-1,1]
    lidar_obs = policy_obs[:, :80]
    assert (lidar_obs >= 0).all() and (lidar_obs <= 1).all(), \
        f"LiDAR 범위 오류: min={lidar_obs.min():.3f}, max={lidar_obs.max():.3f}"

    for _ in range(10):
        act = torch.zeros(env.unwrapped.num_envs, 2)
        obs, rew, term, trunc, _ = env.step(act)

    print(f"reward sample : {rew[:3]}")                # 기대: 음수(time_penalty 포함)
    print(f"obs 82D check : {obs['policy'].shape[-1] == 82}")
    print(f"\n[PASS] 기본 동작 검증 완료.")
    print(f"{'='*50}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
