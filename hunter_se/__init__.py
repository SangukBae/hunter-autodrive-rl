# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE robot USD asset (URDF→USD 변환본).

원본 URDF:
    /robot_isaac/ugv_gazebo_sim/hunter_se/hunter_se_description/urdf/hunter_se_description.urdf

변환 도구:
    urdf-usd-converter v0.1.0

주요 파일:
    hunter_se_description.usda  — Asset Interface (메인 진입점)
    Payload/Contents.usda       — 링크/조인트 계층
    Payload/Physics.usda        — 물리 속성
    Payload/Geometry.usda       — 시각 메시
    Payload/Materials.usda      — 재질
"""

from .hunter_se_cfg import HUNTER_SE_CFG

__all__ = ["HUNTER_SE_CFG"]
