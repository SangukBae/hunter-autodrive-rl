# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR 기반 자율주행 태스크 — goal-reaching + obstacle avoidance.

등록 환경:
    Isaac-LidarNav-Hunter-v0          Phase C (학습, 64 envs)
    Isaac-LidarNav-Hunter-Play-v0     Phase C (시각화, 4 envs)
    Isaac-LidarNav-Hunter-PhaseE-v0   Phase E 물리 벽+장애물 (학습, 64 envs)
    Isaac-LidarNav-Hunter-PhaseE-Play-v0  Phase E (시각화, 4 envs)
"""

import gymnasium as gym

# ── Phase C (기본) ────────────────────────────────────────────────────────────

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

# ── Phase E (물리 벽 + 물리 장애물) ─────────────────────────────────────────

gym.register(
    id="Isaac-LidarNav-Hunter-PhaseE-v0",
    entry_point=f"{__name__}.lidar_nav_env:LidarNavEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_nav_env_cfg:LidarNavEnvCfgPhaseE",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LidarNav-Hunter-PhaseE-Play-v0",
    entry_point=f"{__name__}.lidar_nav_env:LidarNavEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_nav_env_cfg:LidarNavEnvCfgPhaseEPlay",
    },
    disable_env_checker=True,
)
