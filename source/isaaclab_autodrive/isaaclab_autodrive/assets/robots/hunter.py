# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 로봇 ArticulationCfg.

hunter_se/ 폴더의 검증 완료 물리 모델을 참조합니다.

Hunter SE 제원 (Physics.usda 기준):
    - 조향 방식   : Ackermann 전륜 조향
    - 축거        : 0.548 m
    - 윤거 (후륜) : 0.504 m
    - 최대 조향각 : ±22° = ±0.384 rad
    - 최고 속도   : 4.8 km/h (PDF)
    - 자체 중량   : 42 kg (PDF)

관절 구조:
    후륜 구동 (velocity 제어): re_left_joint, re_right_joint
    전륜 조향 (position 제어): fr_steer_left_joint, fr_steer_right_joint
    가상 조향/후륜축 (잠금)  : front_steer_joint, rear_wheel_joint
    전륜 자유회전            : fr_left_joint, fr_right_joint
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Hunter SE USD 파일 경로 (hunter_se/ 검증 완료 모델)
##
HUNTER_SE_USD_PATH = os.path.join(
    "/workspace/hunter_autodrive/hunter_se",
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
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["re_left_joint", "re_right_joint"],
            stiffness=0.0,
            damping=17453.0,
        ),
        # 전륜 조향 위치 제어 — ±22° (0.384 rad) 제한
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

# 하위 호환성 별칭 (legacy 코드에서 HUNTER_CFG를 참조하는 경우 대비)
HUNTER_CFG = HUNTER_SE_CFG
