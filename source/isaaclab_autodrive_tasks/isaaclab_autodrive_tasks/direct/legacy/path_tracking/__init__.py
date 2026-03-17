# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""평탄 지형 경로 추종 태스크."""

import gymnasium as gym

gym.register(
    id="Isaac-PathTracking-Hunter-v0",
    entry_point=f"{__name__}.path_tracking_env:HunterPathTrackingEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.path_tracking_env_cfg:HunterPathTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HunterPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PathTracking-Hunter-Play-v0",
    entry_point=f"{__name__}.path_tracking_env:HunterPathTrackingEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.path_tracking_env_cfg:HunterPathTrackingEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HunterPPORunnerCfg",
    },
    disable_env_checker=True,
)
