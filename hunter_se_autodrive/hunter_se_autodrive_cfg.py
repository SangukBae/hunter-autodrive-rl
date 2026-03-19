# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE (AutoDRIVE 버전) Isaac Lab ArticulationCfg.

USD 원본:
    hunter_se_autodrive_description.usda  (USD 프리미티브 기반)

Hunter SE AutoDRIVE 제원 (HunterSESceneBuilder.cs 기준, 1:5 스케일):
    - 조향 방식   : Ackermann 전륜 조향 (CAWD)
    - 축거        : L = 0.550 m
    - 윤거 (전/후): = 0.520 m
    - 바퀴 반경   : 0.082 m
    - 최대 조향각 : ±30° = ±0.5236 rad
    - 최고 속도   : 3.5611 m/s
    - 자체 중량   : 25 kg

조인트 구조:
    후륜 구동 (velocity 제어):
        re_left_joint   — 후륜 좌
        re_right_joint  — 후륜 우

    전륜 조향 (position 제어):
        fr_steer_left_joint   — 좌 조향 (limits ±30°)
        fr_steer_right_joint  — 우 조향

    가상 조향/보조 (미사용 / 잠금):
        front_steer_joint  — 가상 중앙 조향축
        rear_wheel_joint   — 가상 후륜축

    전륜 자유회전 (front_wheels actuator):
        fr_left_joint   — 전륜 좌 자유회전
        fr_right_joint  — 전륜 우 자유회전
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Hunter SE AutoDRIVE USD 파일 경로
##
HUNTER_SE_AUTODRIVE_USD_PATH = os.path.join(
    os.path.dirname(__file__),
    "hunter_se_autodrive_description.usda",
)

##
# Hunter SE AutoDRIVE ArticulationCfg
##
HUNTER_SE_AUTODRIVE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HUNTER_SE_AUTODRIVE_USD_PATH,
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
        pos=(0.0, 0.0, 0.20),  # 바퀴 반경(0.082) + 여유(0.118) = 0.20m
    ),
    actuators={
        # 후륜 속도 제어 — stiffness=0 (velocity mode)
        # AutoDRIVE MotorTorque=50 N-m 기준, damping 유지
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["re_left_joint", "re_right_joint"],
            stiffness=0.0,
            damping=17453.0,
        ),
        # 전륜 조향 위치 제어 — ±30° (0.5236 rad) 제한
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
        # 전륜 자유회전 — 베어링 마찰 수준
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=["fr_left_joint", "fr_right_joint"],
            stiffness=0.0,
            damping=0.5,
        ),
    },
)
