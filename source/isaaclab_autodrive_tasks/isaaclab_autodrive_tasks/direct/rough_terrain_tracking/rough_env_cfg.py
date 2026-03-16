# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 험로 경로 추종 환경 설정.

path_tracking 환경을 상속하여 험로 지형과 도메인 랜덤화를 추가합니다.
"""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_autodrive_tasks.direct.path_tracking.path_tracking_env_cfg import HunterPathTrackingEnvCfg


@configclass
class HunterRoughTerrainEnvCfg(HunterPathTrackingEnvCfg):
    """험로 경로 추종 환경 설정.

    평탄 환경(HunterPathTrackingEnvCfg)과 동일하나 아래 항목이 추가됩니다:
    - 관측 노이즈 (IMU 시뮬레이션)
    - 도메인 랜덤화 (마찰 계수 변화)
    - 에피소드 길이 단축 (험로로 인한 조기 종료 빈도 증가 대응)
    """

    # ── 씬: 환경 수 줄여서 험로에서도 안정적으로 학습 ───────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=8.0,        # 험로 지형이 넓어서 간격 확대
        replicate_physics=True,
    )

    # ── 에피소드 ────────────────────────────────────────────────────────────────
    episode_length_s: float = 150.0   # 험로는 속도가 느려 150s로 단축

    # ── 종료 조건 완화 (험로는 흔들림이 심함) ─────────────────────────────────
    max_crosstrack_error: float = 6.0  # 5.0 → 6.0m

    # ── 관측 노이즈 모델 (IMU 노이즈 시뮬레이션) ──────────────────────────────
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # ── 행동 노이즈 모델 (액추에이터 지연/오차 시뮬레이션) ──────────────────────
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )


@configclass
class HunterRoughTerrainEnvCfgPlay(HunterRoughTerrainEnvCfg):
    """시각화용 험로 환경 설정."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=8.0,
        replicate_physics=True,
    )
