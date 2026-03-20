# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 Ackermann 조향 처리 모듈.

hunter_se/ackermann.py 와 동일한 수식을 사용.
물리 상수도 동일 (URDF 검증값 기준).

사용 예:
    acker = HunterSEAckermann(device="cuda")
    steer_l, steer_r, omega_l, omega_r = acker.compute(lin_vel, delta_c)

    acker = HunterSEAckermann(device="cpu")
    sl, sr, wl, wr = acker.compute_scalar(v=1.0, delta_c=0.3)
"""

from __future__ import annotations

import torch

# ── Hunter SE 물리 상수 ────────────────────────────────────────────────────────
WHEELBASE    = 0.548    # 축거 L [m]   (FRONT_AX_X − REAR_AX_X = 0.34058+0.2078, URDF 관절 기준)
FRONT_TRACK  = 0.492    # 전륜 피벗 간격 W_f [m]  (FRONT_AX_Y × 2)
REAR_TRACK   = 0.504    # 후륜 허브 간격 W_r [m]  (REAR_AX_Y × 2)
WHEEL_RADIUS = 0.1375   # 바퀴 반지름 r [m]  (매뉴얼 직경 0.275 m / 2)
MAX_STEER    = 0.384    # 최대 차체 중심 조향각 [rad]  (≈ 22°)
MAX_SPEED    = 1.333    # 최대 선속도 [m/s]  (4.8 km/h)

_EPS = 1e-4             # 직진 판정 임계값


class HunterSEAckermann:
    """Hunter SE V0 Ackermann 조향 계산기.

    입력 : 차체 중심(bicycle model) 조향각 delta_c [rad], 선속도 v [m/s]
    출력 : 좌/우 전륜 조향각, 좌/우 후륜 각속도

    수식 (중심각 기준 Ackermann 기하학):
        R        = L / tan(delta_c)
        δ_left   = atan(L / (R − W_f/2))
        δ_right  = atan(L / (R + W_f/2))
        ω_left   = v × (R − W_r/2) / (R × r)
        ω_right  = v × (R + W_r/2) / (R × r)

    부호 규약 (v0 USD axis=Y 기준):
        양수 delta_c = 좌회전
        양수 lin_vel + 양수 omega = 전진
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = device
        self.L   = torch.tensor(WHEELBASE,    device=device, dtype=torch.float32)
        self.W_f = torch.tensor(FRONT_TRACK,  device=device, dtype=torch.float32)
        self.W_r = torch.tensor(REAR_TRACK,   device=device, dtype=torch.float32)
        self.r   = torch.tensor(WHEEL_RADIUS, device=device, dtype=torch.float32)

    def compute(
        self,
        lin_vel: torch.Tensor,   # (N,) 선속도 [m/s]
        delta_c: torch.Tensor,   # (N,) 차체 중심 조향각 [rad]  양수=좌회전
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """배치 텐서 입력으로 관절 목표값 계산."""
        tan_dc   = torch.tan(delta_c)
        straight = tan_dc.abs() < _EPS

        R = torch.where(straight, torch.ones_like(tan_dc), self.L / tan_dc)

        half_wf = self.W_f / 2.0
        steer_l = torch.where(straight, torch.zeros_like(delta_c),
                               torch.atan(self.L / (R - half_wf)))
        steer_r = torch.where(straight, torch.zeros_like(delta_c),
                               torch.atan(self.L / (R + half_wf)))

        half_wr      = self.W_r / 2.0
        omega_straight = lin_vel / self.r
        omega_l = torch.where(
            straight, omega_straight,
            lin_vel * (R - half_wr) / (R * self.r),
        )
        omega_r = torch.where(
            straight, omega_straight,
            lin_vel * (R + half_wr) / (R * self.r),
        )
        return steer_l, steer_r, omega_l, omega_r

    def compute_scalar(
        self, v: float, delta_c: float
    ) -> tuple[float, float, float, float]:
        """스칼라 입력 래퍼 (standalone 스크립트용)."""
        v_t  = torch.tensor([v],       device=self.device, dtype=torch.float32)
        dc_t = torch.tensor([delta_c], device=self.device, dtype=torch.float32)
        sl, sr, wl, wr = self.compute(v_t, dc_t)
        return sl.item(), sr.item(), wl.item(), wr.item()
