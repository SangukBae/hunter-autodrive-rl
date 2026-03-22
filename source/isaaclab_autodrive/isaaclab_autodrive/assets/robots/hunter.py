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

액추에이터 모델 (WheeledLab MuSHR 방식):
    후륜 구동 : DCMotorCfg (토크-속도 포화 커브)
    전륜 조향 : ImplicitActuatorCfg 저강성 (서보 유연성 반영)

바퀴 충돌 근사: convexHull (Physics.usda)
접지 충격 제한: max_contact_impulse=0.0
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
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
        pos=(0.0, 0.0, 0.3),  # 지형 위 안전 높이
    ),
    actuators={
        # 후륜 구동 — DCMotorCfg (MuSHR throttle 방식)
        "wheels": DCMotorCfg(
            joint_names_expr=["re_left_joint", "re_right_joint"],
            saturation_effort=30.0,
            effort_limit=15.0,
            velocity_limit=12.0,
            stiffness=0.0,
            damping=30.0,
            friction=0.0,
        ),
        # 전륜 조향 — 저강성 ImplicitActuatorCfg (MuSHR steering 방식 스케일)
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["fr_steer_left_joint", "fr_steer_right_joint"],
            stiffness=500.0,
            damping=50.0,
            effort_limit_sim=50.0,
        ),
        # 가상 조향/후륜축 — 고강성 잠금
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

# ── Hunter SE V0 (기본 도형 기반, 검증된 물리 파라미터) ──────────────────────
# hunter_se_v0_cfg.py 에 정의된 CFG를 그대로 참조.
# 주요 차이점:
#   - 바퀴 반지름: 0.129 → 0.1375 m  (공식 메뉴얼 기준)
#   - virtual_joints 없음  (v0 단순화)
#   - DCMotorCfg damping=15 (50Hz 이산 시간 안정성 최적화)
#   - velocity_limit=15 rad/s (크루즈 속도 포화 토크 여유 확보)
#   - 전륜 damping=0.01 (실제 베어링 마찰 수준, 전진 저항 최소화)
import sys as _sys
_sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_v0.hunter_se_v0_cfg import HUNTER_SE_V0_CFG  # noqa: E402
