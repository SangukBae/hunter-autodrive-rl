# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""다중 트랙 일반화 태스크 (Phase 4에서 구현 예정)."""

import gymnasium as gym

gym.register(
    id="Isaac-MultiTrack-Hunter-v0",
    entry_point="isaaclab_autodrive_tasks.direct.multi_track.multi_track_env:HunterMultiTrackEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_autodrive_tasks.direct.multi_track.multi_track_env_cfg:HunterMultiTrackEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_autodrive_tasks.direct.multi_track.agents.rsl_rl_ppo_cfg:HunterMultiTrackPPORunnerCfg",
    },
    disable_env_checker=True,
)
