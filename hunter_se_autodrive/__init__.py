# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE (AutoDRIVE 시뮬레이션 버전) USD 에셋.

원본 소스:
    AutoDRIVE-Simulator-0.3.0/Assets/Models/Vehicle/Hunter SE/Hunter SE.fbx
    AutoDRIVE-Simulator-0.3.0/Assets/Editor/HunterSESceneBuilder.cs

물리 파라미터 출처:
    HunterSESceneBuilder.cs (AutoDRIVE 1:5 스케일 모델)

주요 파일:
    hunter_se_autodrive_description.usda  — Asset Interface (메인 진입점)
    Payload/Contents.usda                 — 링크/조인트 계층
    Payload/Physics.usda                  — 물리 속성
    Payload/Geometry.usda                 — 시각 메시 (USD 프리미티브)
    Payload/Materials.usda                — 재질
"""

from .hunter_se_autodrive_cfg import HUNTER_SE_AUTODRIVE_CFG

__all__ = ["HUNTER_SE_AUTODRIVE_CFG"]
