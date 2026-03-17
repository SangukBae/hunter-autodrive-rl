# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 험로 경로 추종 RSL-RL PPO 에이전트 설정."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HunterRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """험로 환경용 RSL-RL PPO — 평탄 환경보다 더 많은 이터레이션."""

    num_steps_per_env: int = 24         # 험로는 롤아웃을 길게
    max_iterations: int = 5000
    save_interval: int = 200
    experiment_name: str = "hunter_rough_terrain_tracking"
    empirical_normalization: bool = False

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,          # 험로는 탐색을 더 늘림
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
