# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE Ackermann 조향 처리 모듈.

WheeledLab AckermannAction의 설계를 참고하여 구현:
    - 입력: 차체 중심(bicyle model) 조향각 delta_c [rad], 선속도 v [m/s]
    - 출력: 좌/우 전륜 조향각, 좌/우 후륜 각속도 (모두 GPU 텐서)

WheeledLab과의 차이점:
    - DirectRLEnv 기반이므로 ActionTerm 상속 대신 독립 클래스로 구현
    - 후륜 2WD (전륜 자유회전) — 전/후륜 모두 구동하는 WheeledLab F1Tenth와 달리
      MuSHR RWD 방식으로 후륜만 구동
    - 조향각도 기준: 차체 중심 조향각 (WheeledLab AckermannAction과 동일)

수식 (Ackermann 기하학, 중심각 기준):
    R       = L / tan(delta_c)                  # 회전반경 (후륜 중심 기준)
    δ_left  = atan(L / (R − W_f/2))             # 좌 전륜각 (좌회전 시 내륜 = 큰 각도)
    δ_right = atan(L / (R + W_f/2))             # 우 전륜각 (좌회전 시 외륜 = 작은 각도)
    ω_left  = v × (R − W_r/2) / (R × r)        # 좌 후륜 각속도 (좌회전 시 내륜 = 느림)
    ω_right = v × (R + W_r/2) / (R × r)        # 우 후륜 각속도 (좌회전 시 외륜 = 빠름)

    직진(delta_c ≈ 0): R → ∞, δ_left = δ_right = 0, ω_left = ω_right = v/r

사용 예:
    # 배치 텐서 모드 (RL 환경)
    acker = HunterSEAckermann(device="cuda")
    steer_l, steer_r, omega_l, omega_r = acker.compute(lin_vel, delta_c)

    # 스칼라 모드 (standalone 스크립트)
    acker = HunterSEAckermann(device="cpu")
    sl, sr, wl, wr = acker.compute_scalar(v=1.0, delta_c=0.3)
"""

from __future__ import annotations

import math

import torch

# ── Hunter SE 물리 상수 ────────────────────────────────────────────────────────
WHEELBASE    = 0.548   # 축거 L [m]
FRONT_TRACK  = 0.492   # 전륜 조향 피벗 간격 W_f [m]  (±0.24619 × 2)
REAR_TRACK   = 0.504   # 후륜 허브 간격 W_r [m]       (±0.252 × 2)
WHEEL_RADIUS = 0.129   # 바퀴 반지름 r [m]
MAX_STEER    = 0.384   # 최대 내륜 조향각 [rad] (±22°)
MAX_SPEED    = 1.333   # 최대 선속도 [m/s] (4.8 km/h)

# 직진 판정 임계값 (|delta_c| < EPS → 직진 처리)
_EPS = 1e-4


class HunterSEAckermann:
    """Hunter SE Ackermann 조향 계산기 (GPU 텐서 기반).

    WheeledLab AckermannAction._calculate_ackermann_angles_and_velocities()와
    동일한 중심각 기준 공식을 사용. DirectRLEnv에서 직접 호출 가능하도록
    ActionTerm 상속 없이 독립 클래스로 구현.

    Attributes:
        device: 텐서 연산 디바이스.
        L: 축거 [m].
        W_f: 전륜 조향 피벗 간격 [m].
        W_r: 후륜 허브 간격 [m].
        r: 바퀴 반지름 [m].
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = device
        self.L   = torch.tensor(WHEELBASE,    device=device, dtype=torch.float32)
        self.W_f = torch.tensor(FRONT_TRACK,  device=device, dtype=torch.float32)
        self.W_r = torch.tensor(REAR_TRACK,   device=device, dtype=torch.float32)
        self.r   = torch.tensor(WHEEL_RADIUS, device=device, dtype=torch.float32)

    # ── 배치 텐서 연산 (RL 환경용) ────────────────────────────────────────────

    def compute(
        self,
        lin_vel: torch.Tensor,   # (N,) 선속도 [m/s]
        delta_c: torch.Tensor,   # (N,) 차체 중심 조향각 [rad]  양수=좌회전
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """배치 텐서 입력으로 관절 목표값 계산.

        Args:
            lin_vel: 선속도 (N,) [m/s]. 양수=전진.
            delta_c: 차체 중심 조향각 (N,) [rad]. 양수=좌회전.

        Returns:
            (steer_l, steer_r, omega_l, omega_r) 모두 (N,) 텐서.
                steer_l/r : 전륜 조향 관절 position target [rad]
                omega_l/r : 후륜 velocity target [rad/s]
        """
        L   = self.L
        W_f = self.W_f
        W_r = self.W_r
        r   = self.r

        tan_dc = torch.tan(delta_c)

        # 직진 분기: |delta_c| < EPS → R → ∞ 특이점 방지
        straight = tan_dc.abs() < _EPS

        # 회전반경 (0으로 나누기 방지: 직진 구간은 dummy 값 1.0 사용 후 마스킹)
        R = torch.where(
            straight,
            torch.ones_like(tan_dc),   # dummy (직진 마스킹됨)
            L / tan_dc,
        )

        # ── 전륜 조향각 (WheeledLab AckermannAction과 동일 공식) ──────────────
        # delta_left  = atan(L / (R - W_f/2))
        # delta_right = atan(L / (R + W_f/2))
        half_wf = W_f / 2.0
        steer_l = torch.where(straight, torch.zeros_like(delta_c),
                              torch.atan(L / (R - half_wf)))
        steer_r = torch.where(straight, torch.zeros_like(delta_c),
                              torch.atan(L / (R + half_wf)))

        # ── 후륜 차동 각속도 (MuSHR RWD 방식: 후륜만 구동) ──────────────────
        half_wr = W_r / 2.0
        omega_straight = lin_vel / r

        omega_l = torch.where(
            straight,
            omega_straight,
            lin_vel * (R - half_wr) / (R * r),
        )
        omega_r = torch.where(
            straight,
            omega_straight,
            lin_vel * (R + half_wr) / (R * r),
        )

        return steer_l, steer_r, omega_l, omega_r

    # ── 스칼라 래퍼 (standalone 스크립트용) ──────────────────────────────────

    def compute_scalar(
        self, v: float, delta_c: float
    ) -> tuple[float, float, float, float]:
        """스칼라 입력 래퍼. compute()를 (1,) 텐서로 호출 후 float 반환.

        Args:
            v:       선속도 [m/s].
            delta_c: 차체 중심 조향각 [rad]. 양수=좌회전.

        Returns:
            (steer_l, steer_r, omega_l, omega_r) float.
        """
        v_t  = torch.tensor([v],       device=self.device, dtype=torch.float32)
        dc_t = torch.tensor([delta_c], device=self.device, dtype=torch.float32)
        sl, sr, wl, wr = self.compute(v_t, dc_t)
        return sl.item(), sr.item(), wl.item(), wr.item()
