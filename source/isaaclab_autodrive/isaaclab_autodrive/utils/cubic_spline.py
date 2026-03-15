# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""3차 스플라인 경로 생성 유틸리티.

원본 출처:
    Hybrid_DRL_Deployments/.../hunter_hybrid/CubicSpline.py
    Author: Atsushi Sakai (@Atsushi_twi)

사용 예시:
    >>> coords = np.genfromtxt('austin_centerline2.csv', delimiter=',')
    >>> x, y = coords[::10, 0], coords[::10, 1]
    >>> cx, cy, cyaw, ck, s = calc_spline_course(x, y, ds=0.1)
"""

from __future__ import annotations

import bisect
import math

import numpy as np


class CubicSpline1D:
    """1D 자연 3차 스플라인 보간.

    Args:
        x: 오름차순 정렬된 x 좌표 배열
        y: y 좌표 배열

    Raises:
        ValueError: x 가 오름차순 정렬되지 않은 경우
    """

    def __init__(self, x: list | np.ndarray, y: list | np.ndarray):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x 좌표는 오름차순으로 정렬되어야 합니다.")

        self.x = list(x)
        self.y = list(y)
        self.nx = len(x)
        self.a = list(y)

        A = self._calc_A(h)
        B = self._calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        self.b, self.d = [], []
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = (self.a[i + 1] - self.a[i]) / h[i] - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x: float) -> float | None:
        """주어진 x 에서 y 위치 계산. 범위 밖이면 None 반환."""
        if x < self.x[0] or x > self.x[-1]:
            return None
        i = self._search_index(x)
        dx = x - self.x[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3

    def calc_first_derivative(self, x: float) -> float | None:
        """1차 도함수 계산. 범위 밖이면 None 반환."""
        if x < self.x[0] or x > self.x[-1]:
            return None
        i = self._search_index(x)
        dx = x - self.x[i]
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2

    def calc_second_derivative(self, x: float) -> float | None:
        """2차 도함수 계산. 범위 밖이면 None 반환."""
        if x < self.x[0] or x > self.x[-1]:
            return None
        i = self._search_index(x)
        dx = x - self.x[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

    def _search_index(self, x: float) -> int:
        return bisect.bisect(self.x, x) - 1

    def _calc_A(self, h: np.ndarray) -> np.ndarray:
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        A[self.nx - 1, self.nx - 1] = 1.0
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        for i in range(self.nx - 1):
            if i != self.nx - 2:
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
        return A

    def _calc_B(self, h: np.ndarray, a: list) -> np.ndarray:
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:
    """2D 3차 스플라인 경로.

    호 길이(arc length) 기반으로 x, y 를 독립적으로 보간합니다.

    Args:
        x: x 좌표 배열
        y: y 좌표 배열
    """

    def __init__(self, x: list | np.ndarray, y: list | np.ndarray):
        self.s = self._calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def _calc_s(self, x: list | np.ndarray, y: list | np.ndarray) -> list:
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s: float) -> tuple[float | None, float | None]:
        """호 길이 s 에서 (x, y) 위치 계산."""
        return self.sx.calc_position(s), self.sy.calc_position(s)

    def calc_curvature(self, s: float) -> float | None:
        """호 길이 s 에서 곡률 계산."""
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        if dx is None or dy is None or ddx is None or ddy is None:
            return None
        return (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** 1.5)

    def calc_yaw(self, s: float) -> float | None:
        """호 길이 s 에서 yaw 각도(접선 방향) 계산 [rad]."""
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        if dx is None or dy is None:
            return None
        return math.atan2(dy, dx)


def calc_spline_course(
    x: list | np.ndarray,
    y: list | np.ndarray,
    ds: float = 0.1,
) -> tuple[list, list, list, list, list]:
    """경로점 배열로부터 3차 스플라인 코스를 생성합니다.

    Args:
        x: x 좌표 배열 (경로점)
        y: y 좌표 배열 (경로점)
        ds: 보간 간격 [m] (기본값: 0.1m)

    Returns:
        (cx, cy, cyaw, ck, s) 튜플:
            cx: 보간된 x 좌표 리스트
            cy: 보간된 y 좌표 리스트
            cyaw: 각 점에서의 yaw 각도 리스트 [rad]
            ck: 각 점에서의 곡률 리스트 [1/m]
            s: 호 길이 리스트 [m]
    """
    sp = CubicSpline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    cx, cy, cyaw, ck = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        cx.append(ix)
        cy.append(iy)
        cyaw.append(sp.calc_yaw(i_s))
        ck.append(sp.calc_curvature(i_s))

    return cx, cy, cyaw, ck, s
