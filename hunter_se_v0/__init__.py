# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 — 기본 도형(Box/Cylinder) 기반 procedural USD 로봇 패키지.

hunter_se/ 폴더(URDF→USD 변환본)와 달리, 이 패키지는 pxr Python API로
기본 도형만 사용하여 USD articulation을 동적으로 생성합니다.

물리 파라미터(관절 위치·방향·질량)는 기존 hunter_se/ URDF에서 추출한
검증된 값을 그대로 사용합니다.
"""
