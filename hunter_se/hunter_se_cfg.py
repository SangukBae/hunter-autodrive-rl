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
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),  # 지형 위 안전 높이
    ),
    actuators={
        # 후륜 속도 제어 — stiffness=0 (velocity mode)
        # hunter_aim4 참고: damping=17453, maxForce=inf
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["re_left_joint", "re_right_joint"],
            stiffness=0.0,
            damping=17453.0,
        ),
        # 전륜 조향 위치 제어 — ±22° (0.384 rad) 제한
        # hunter_aim4 참고: stiffness=10M, damping=100K, maxForce=6000
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["fr_steer_left_joint", "fr_steer_right_joint"],
            stiffness=1e7,
            damping=1e5,
            effort_limit_sim=6000.0,
        ),
        # 가상 조향/후륜축 — 고강성으로 잠금
        "virtual_joints": ImplicitActuatorCfg(
            joint_names_expr=["front_steer_joint", "rear_wheel_joint"],
            stiffness=1e7,
            damping=1e5,
            effort_limit_sim=1e6,
        ),
        # 전륜 자유회전 — DOF축이 구름 방향이므로 stiffness=0 필수
        # damping=15 원본값은 URDF friction=15(쿨롱)가 점성으로 잘못 변환된 것
        # → 7.75 rad/s에서 116 Nm 제동(법력 82 N 대비 900 N 필요) = 사실상 브레이크
        # damping=0.5로 수정: 베어링 마찰 수준 (≈ 4 Nm @ 7.75 rad/s)
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=["fr_left_joint", "fr_right_joint"],
            stiffness=0.0,
            damping=0.5,
        ),
    },
)
