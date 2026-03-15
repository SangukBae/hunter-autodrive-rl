# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""각도 유틸리티 함수.

원본 출처:
    Hybrid_DRL_Deployments/.../hunter_hybrid/angle.py
    Author: Atsushi Sakai (@Atsushi_twi)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle: float) -> np.ndarray:
    """주어진 각도로 2D 회전 행렬을 생성합니다.

    Args:
        angle: 회전 각도 [rad]

    Returns:
        shape (2, 2) 의 2D 회전 행렬
    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def angle_mod(
    x: float | np.ndarray,
    zero_2_2pi: bool = False,
    degree: bool = False,
) -> float | np.ndarray:
    """각도 모듈로 연산.

    기본 범위: [-pi, pi)
    zero_2_2pi=True 시 범위: [0, 2pi)

    Args:
        x: 각도 또는 각도 배열 [rad] (degree=True 이면 [deg])
        zero_2_2pi: True 이면 [0, 2pi) 범위로 변환
        degree: True 이면 입력을 degree 로 해석

    Returns:
        모듈로 연산된 각도 (입력이 float 이면 float, 배열이면 ndarray)

    Examples:
        >>> angle_mod(-4.0)
        2.2831853...
        >>> angle_mod(-150.0, degree=True)
        -150.0
    """
    is_float = isinstance(x, float)
    x = np.asarray(x, dtype=float).flatten()

    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    return float(mod_angle.item()) if is_float else mod_angle
