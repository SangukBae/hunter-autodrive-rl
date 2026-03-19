# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE Isaac Lab ArticulationCfg (URDF→USD 변환본 기반).

USD 원본:
    hunter_se_description.usda  (urdf-usd-converter v0.1.0 생성)

Hunter SE 제원 (URDF + PDF 매뉴얼 기준):
    - 조향 방식   : Ackermann 전륜 조향
    - 축거        : L ≈ 0.548 m  (URDF 조인트 위치 기준)
    - 윤거 (후륜) : ≈ 0.504 m  (URDF 조인트 위치 기준)
    - 최대 조향각 : ±22° = ±0.384 rad  (PDF: Max inner wheel steering angle)
    - 최고 속도   : 4.8 km/h  (PDF)
    - 자체 중량   : 42 kg  (PDF)
    - 지상고      : 120 mm  (PDF)

조인트 구조:
    후륜 구동 (velocity 제어):
        re_left_joint   — 후륜 좌
        re_right_joint  — 후륜 우

    전륜 조향 (position 제어):
        fr_steer_left_joint   — 좌 조향 (limits ±39.5° in USD, 실제 ±22° 적용)
        fr_steer_right_joint  — 우 조향

    가상 조향/보조 (미사용 / 잠금):
        front_steer_joint  — 가상 중앙 조향축
        rear_wheel_joint   — 가상 후륜축

    전륜 자유회전 (front_wheels actuator, 마찰=15):
        fr_left_joint   — 전륜 좌 자유회전
        fr_right_joint  — 전륜 우 자유회전

액추에이터 모델 (WheeledLab MuSHR 방식 적용):
    후륜 구동  : DCMotorCfg — 토크-속도 포화 커브 모델링
    전륜 조향  : ImplicitActuatorCfg — 저강성 서보 (MuSHR 스케일 적용)

바퀴 충돌 근사:
    boundingSphere → convexHull (Physics.usda 변경)
    — 바퀴 형상을 구(球)에서 볼록 껍질로 개선, 지면 접지 정확도 향상

접지 충격 제한:
    max_contact_impulse=0.0 (PhysX 기본 = 무제한)
    — MuSHR과 동일하게 설정하여 접지 충격 클램핑 해제
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Hunter SE USD 파일 경로 (변환본)
##
HUNTER_SE_USD_PATH = os.path.join(
    os.path.dirname(__file__),
    "hunter_se_description.usda",
)

##
# Hunter SE ArticulationCfg
##
HUNTER_SE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HUNTER_SE_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            max_contact_impulse=0.0,       # MuSHR 방식: 접지 충격 클램핑 해제
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.287),  # 후륜축 z=-0.158 + 바퀴반지름 0.129 = 정지 높이
    ),
    actuators={
        # ── 후륜 구동 — DCMotorCfg (MuSHR throttle 방식) ─────────────────────
        # torque-speed 포화 커브: τ_eff = clip(kd·Δω, ±τ_sat) × (1 − |ω|/ω_max)
        # Hunter SE 추정치:
        #   τ_sat = 30 N·m  (후륜 1개당 정지 토크, 42 kg 기준)
        #   ω_max = 12 rad/s (> 운전 최대 10.33 rad/s)
        #   damping = 30  → 정상상태 속도 오차 ≈ 3% (안정 한계 ~70 N·m·s/rad의 43%)
        #   전체 로봇 유효 관성: I_eff = r²·m = 0.129²·42 = 0.699 kg·m²
        #   50Hz 제어 안정 한계: kd < 2·I_eff/STEP_DT = 2·0.699/0.02 ≈ 70
        "wheels": DCMotorCfg(
            joint_names_expr=["re_left_joint", "re_right_joint"],
            saturation_effort=30.0,
            effort_limit=15.0,
            velocity_limit=12.0,
            stiffness=0.0,
            damping=30.0,
            friction=0.0,
        ),
        # ── 전륜 조향 — ImplicitActuatorCfg 저강성 (MuSHR steering 방식) ──────
        # MuSHR: stiffness=100, damping=10, effort_limit=3.2
        # Hunter SE: 42 kg / ~5 kg ≈ 8.4× 스케일
        #   stiffness=500, damping=50, effort_limit=50
        # → 강체 잠금(1e7) 대신 실제 서보 모터의 유연성 반영
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["fr_steer_left_joint", "fr_steer_right_joint"],
            stiffness=500.0,
            damping=50.0,
            effort_limit_sim=50.0,
        ),
        # ── 가상 조향/후륜축 — 고강성으로 잠금 (변경 없음) ───────────────────
        "virtual_joints": ImplicitActuatorCfg(
            joint_names_expr=["front_steer_joint", "rear_wheel_joint"],
            stiffness=1e7,
            damping=1e5,
            effort_limit_sim=1e6,
        ),
        # ── 전륜 자유회전 — 베어링 마찰 수준 (변경 없음) ─────────────────────
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=["fr_left_joint", "fr_right_joint"],
            stiffness=0.0,
            damping=0.5,
        ),
    },
)
