# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR 기반 자율주행 태스크 — goal-reaching + obstacle avoidance.

등록 환경:
    Isaac-LidarNav-Hunter-v0          학습용 (64 envs, RTX LiDAR, 16×16 벽+장애물)
    Isaac-LidarNav-Hunter-Play-v0     시각화용 (4 envs)
"""

import gymnasium as gym

gym.register(
    id="Isaac-LidarNav-Hunter-v0",
    entry_point=f"{__name__}.lidar_nav_env:LidarNavEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_nav_env_cfg:LidarNavEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LidarNav-Hunter-Play-v0",
    entry_point=f"{__name__}.lidar_nav_env:LidarNavEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_nav_env_cfg:LidarNavEnvCfgPlay",
    },
    disable_env_checker=True,
)
