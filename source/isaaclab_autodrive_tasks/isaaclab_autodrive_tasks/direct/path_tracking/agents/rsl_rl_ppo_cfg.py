# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 경로 추종 RSL-RL PPO 에이전트 설정.

원본 출처:
    Hybrid_DRL_Deployments/.../hunter_hybrid/agents/rsl_rl_ppo_hunter_cfg.py
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HunterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO 학습 설정."""

    num_steps_per_env: int = 16         # 환경당 롤아웃 스텝
    max_iterations: int = 3000          # 최대 학습 이터레이션
    save_interval: int = 100            # 체크포인트 저장 간격
    experiment_name: str = "hunter_path_tracking"
    empirical_normalization: bool = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
