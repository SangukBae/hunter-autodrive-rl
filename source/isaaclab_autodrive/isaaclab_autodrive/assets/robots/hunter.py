# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 로봇 ArticulationCfg.

hunter_se_v0/ 폴더의 검증 완료 물리 모델을 참조합니다.

Hunter SE 제원 (공식 메뉴얼 기준):
    - 조향 방식   : Ackermann 전륜 조향
    - 축거        : 0.548 m
    - 윤거 (후륜) : 0.504 m
    - 최대 조향각 : ±22° = ±0.384 rad
    - 최고 속도   : 4.8 km/h
    - 자체 중량   : 42 kg
"""

from __future__ import annotations

import sys as _sys
_sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_v0.hunter_se_v0_cfg import HUNTER_SE_V0_CFG  # noqa: E402

__all__ = ["HUNTER_SE_V0_CFG"]
