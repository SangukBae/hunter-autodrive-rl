# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 로봇 ArticulationCfg.

원본 출처:
    Hybrid_DRL_Deployments/source/isaaclab_assets/isaaclab_assets/robots/hunter.py

Hunter SE 제원:
    - 조향 방식: Ackermann 전륜 조향
    - 축거 (Wheelbase): L = 0.608 m
    - 윤거 (Track width): 0.554 m
    - 최대 조향각: ±0.524 rad (±30°)
    - 구동 방식: 후륜 속도 제어 (re_left_jiont, re_right_jiont)
    - 조향 제어: 전륜 위치 제어 (fr_steer_left_joint, fr_steer_right_joint)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Hunter SE USD 파일 경로
##
HUNTER_USD_PATH = (
    "/robot_isaac/ros2_ws/src/"
    "Hybrid_Deep_Reinforcement_Learning_RoughTerrain/"
    "omniisaacgymenvs/USD_Files/hunter_aim4.usd"
)

##
# Hunter SE ArticulationCfg
##
HUNTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HUNTER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
    ),
    actuators={
        # 후륜 속도 제어 — stiffness=0 (velocity mode)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["re_.*"],
            stiffness=0.0,
            damping=1000.0,
        ),
        # 전륜 조향 위치 제어
        # fr_.* 대신 fr_steer_.* 사용 — fr_left_jiont / fr_right_jiont(전륜 자유회전) 제외
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["fr_steer_.*"],
            stiffness={
                "fr_steer_left_joint": 20.0,
                "fr_steer_right_joint": 20.0,
            },
            damping={
                "fr_steer_left_joint": 0.5,
                "fr_steer_right_joint": 0.5,
            },
        ),
    },
)
