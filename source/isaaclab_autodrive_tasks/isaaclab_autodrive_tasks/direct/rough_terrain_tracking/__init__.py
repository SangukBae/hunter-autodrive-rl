# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""험로 경로 추종 태스크 (Phase 2에서 구현 예정)."""

import gymnasium as gym

gym.register(
    id="Isaac-RoughTerrainTracking-Hunter-v0",
    entry_point="isaaclab_autodrive_tasks.direct.rough_terrain_tracking.rough_env:HunterRoughTerrainEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_autodrive_tasks.direct.rough_terrain_tracking.rough_env_cfg:HunterRoughTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_autodrive_tasks.direct.rough_terrain_tracking.agents.rsl_rl_ppo_cfg:HunterRoughPPORunnerCfg",
    },
    disable_env_checker=True,
)
