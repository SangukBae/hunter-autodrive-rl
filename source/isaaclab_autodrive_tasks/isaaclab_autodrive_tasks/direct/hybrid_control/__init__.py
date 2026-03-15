# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""하이브리드 제어 태스크 DRL + LQR (Phase 4에서 구현 예정)."""

import gymnasium as gym

gym.register(
    id="Isaac-HybridControl-Hunter-v0",
    entry_point="isaaclab_autodrive_tasks.direct.hybrid_control.hybrid_env:HunterHybridEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_autodrive_tasks.direct.hybrid_control.hybrid_env_cfg:HunterHybridEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_autodrive_tasks.direct.hybrid_control.agents.rsl_rl_ppo_cfg:HunterHybridPPORunnerCfg",
    },
    disable_env_checker=True,
)
