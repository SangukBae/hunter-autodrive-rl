# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LQR (Linear Quadratic Regulator) 경로 추종 제어기.

원본 출처:
    Hybrid_DRL_Deployments/.../hunter_hybrid/LQRController.py
    Author: Atsushi Sakai (@Atsushi_twi)

Hunter SE 제원 기반 기본값:
    - 축거: L = 0.550 m  (PDF: Front/rear track 550 mm)
    - 시뮬레이션 dt: 0.005 s
    - LQR Q: diag(1, 10, 100, 100)
    - LQR R: eye(1)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la


class State:
    """차량 상태.

    Attributes:
        x: x 위치 [m]
        y: y 위치 [m]
        yaw: 헤딩 각도 [rad]
        v: 선속도 [m/s]
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, v: float = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


class LQRController:
    """이산 시간 LQR 조향 제어기.

    DARE (Discrete Algebraic Riccati Equation) 를 반복 풀어
    최적 피드백 게인 K 를 구합니다.

    Args:
        L: 축거 [m] (기본값: 0.550)
        dt: 제어 주기 [s] (기본값: 0.005)
        Q: 상태 가중 행렬 shape (4, 4) (기본값: diag(1, 10, 100, 100))
        R: 입력 가중 행렬 shape (1, 1) (기본값: eye(1))
        max_iter: DARE 최대 반복 횟수 (기본값: 150)
        eps: 수렴 판단 임계값 (기본값: 0.0167)
        max_steer: 최대 조향각 [rad] (기본값: 0.384, 22°)
    """

    def __init__(
        self,
        L: float = 0.550,
        dt: float = 0.005,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        max_iter: int = 150,
        eps: float = 0.0167,
        max_steer: float = 0.384,
    ):
        self.L = L
        self.dt = dt
        self.max_iter = max_iter
        self.eps = eps
        self.max_steer = max_steer

        # 기본 Q, R 행렬
        if Q is None:
            self.Q = np.eye(4)
            self.Q[0, 1] = 10.0
            self.Q[1, 2] = 100.0
            self.Q[2, 3] = 100.0
        else:
            self.Q = Q

        self.R = R if R is not None else np.eye(1)

    def calc_steering(
        self,
        state: State,
        crosstrack_error: float,
        heading_error: float,
        curvature: float,
    ) -> float:
        """LQR 피드백 제어로 최적 조향각을 계산합니다.

        Args:
            state: 현재 차량 상태 (x, y, yaw, v)
            crosstrack_error: 경로 횡방향 오차 [m]
            heading_error: 헤딩 오차 [rad]
            curvature: 현재 경로점 곡률 [1/m]

        Returns:
            최적 조향각 [rad], max_steer 범위로 클램프됨
        """
        v = max(state.v, 1e-3)  # 0 나누기 방지

        # 선형화된 상태 방정식 행렬
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = self.dt

        B = np.zeros((4, 1))
        B[3, 0] = v / self.L

        # DARE 반복 풀이
        X = self.Q.copy()
        for _ in range(self.max_iter):
            X_next = (
                A.T @ X @ A
                - A.T @ X @ B @ la.inv(self.R + B.T @ X @ B) @ B.T @ X @ A
                + self.Q
            )
            if np.abs(X_next - X).max() < self.eps:
                break
            X = X_next

        # 피드백 게인
        K = la.inv(B.T @ X @ B + self.R) @ (B.T @ X @ A)

        # 상태 벡터 구성 [e, de/dt, th_e, dth_e/dt]
        x_vec = np.zeros((4, 1))
        x_vec[0, 0] = crosstrack_error
        x_vec[2, 0] = heading_error

        # 피드포워드 (곡률 기반) + 피드백
        ff = np.arctan2(self.L * curvature, 1.0)
        fb = float((-K @ x_vec)[0, 0])
        steer = ff + fb

        return float(np.clip(steer, -self.max_steer, self.max_steer))
