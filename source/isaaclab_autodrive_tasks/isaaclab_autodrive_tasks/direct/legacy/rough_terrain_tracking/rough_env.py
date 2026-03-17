# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 험로 경로 추종 환경.

HunterPathTrackingEnv를 상속하여 험로 지형에서의 주행을 학습합니다.
현재는 평탄 지형 위에서 노이즈만 추가된 형태이며,
추후 terraintrain_9_uneven.usd 지형을 연결할 수 있습니다.
"""

from __future__ import annotations

from isaaclab_autodrive_tasks.direct.path_tracking.path_tracking_env import HunterPathTrackingEnv

from .rough_env_cfg import HunterRoughTerrainEnvCfg


class HunterRoughTerrainEnv(HunterPathTrackingEnv):
    """험로 경로 추종 환경.

    HunterPathTrackingEnv와 동일한 구조이나
    rough_env_cfg에서 정의된 노이즈 모델이 자동으로 적용됩니다.
    (DirectRLEnv가 observation_noise_model, action_noise_model을 자동 처리)
    """

    cfg: HunterRoughTerrainEnvCfg
