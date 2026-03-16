# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""험로 경로 추종 태스크."""

import gymnasium as gym

gym.register(
    id="Isaac-RoughTerrainTracking-Hunter-v0",
    entry_point=f"{__name__}.rough_env:HunterRoughTerrainEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HunterRoughTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HunterRoughPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-RoughTerrainTracking-Hunter-Play-v0",
    entry_point=f"{__name__}.rough_env:HunterRoughTerrainEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HunterRoughTerrainEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HunterRoughPPORunnerCfg",
    },
    disable_env_checker=True,
)
