# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 ArticulationCfg — 기본 도형 기반 procedural USD.

hunter_se/hunter_se_cfg.py 와 동일한 액추에이터 파라미터를 사용하되,
로봇 시각·충돌 형상은 build_usd.py 가 동적으로 생성한 기본 도형(Box/Cylinder) USD를
참조합니다.

차이점:
    - virtual_joints (front_steer_joint, rear_wheel_joint) 제외  (v0 단순화)
    - 바퀴 관절 axis=Y  (원본 axis=Z + localRot 방식과 동일 부호 규약)
    - 관절 이름은 원본과 동일하므로 teleop/RL 스크립트 재사용 가능

관절 이름:
    후륜 구동  : re_left_joint,       re_right_joint
    전륜 조향  : fr_steer_left_joint, fr_steer_right_joint
    전륜 자유  : fr_left_joint,       fr_right_joint
"""

from __future__ import annotations

import os

# ── USD 경로 및 자동 생성 ──────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
HUNTER_SE_V0_USD_PATH = os.path.join(_DIR, "hunter_se_v0.usda")

if not os.path.exists(HUNTER_SE_V0_USD_PATH):
    # Isaac Sim Python 환경에서만 pxr 사용 가능.
    # cfg.py 임포트 시 USD가 없으면 자동 생성.
    from hunter_se_v0.build_usd import build as _build
    _build(HUNTER_SE_V0_USD_PATH)

# ── Isaac Lab 임포트 (Isaac Sim 초기화 이후에 임포트 되어야 함) ────────────────
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Hunter SE V0 ArticulationCfg
##
HUNTER_SE_V0_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HUNTER_SE_V0_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            max_contact_impulse=0.0,          # 접지 충격 클램핑 해제
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
        # 스폰 높이: 바퀴 반지름(0.1375m) + |REAR_AX_Z|(0.158m) = 0.2955m
        pos=(0.0, 0.0, 0.2955),
    ),
    actuators={
        # ── 후륜 구동: DCMotorCfg (토크-속도 포화 커브) ─────────────────────
        # r=0.1375m, 총질량=42kg 기준 후륜 슬립 한계: 101.4N × 0.1375m = 13.94 N·m
        #   velocity_limit=15 rad/s → 크루즈(ω=9.695)에서 포화 토크 여유 확보
        #     τ_sat = 30×(1−9.695/15) = 10.6 N·m  →  drive_force = 77.1 N  < 101.4 N ✓
        #   effort_limit=11.0: 전속력 구간 내내 drive_force=80.0N < 101.4N 유지 (슬립 방지)
        #   damping=15: 50 Hz 제어 루프 이산 시간 안정성
        #     K = 2×d/r² = 1,585 N/(m/s), M_eff=45 kg
        #     이산 극점 = 1 − K×h/M = 1 − 1585×0.02/45 = +0.295  (양수 → 단조 수렴, 진동 없음)
        #     (d=30일 때 극점 = −0.41 → 매 스텝 부호 반전으로 속도 진동 발생)
        "wheels": DCMotorCfg(
            joint_names_expr=["re_left_joint", "re_right_joint"],
            saturation_effort=30.0,
            effort_limit=11.0,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=15.0,
            friction=0.0,
        ),
        # ── 전륜 조향: ImplicitActuatorCfg (저강성 서보) ─────────────────────
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["fr_steer_left_joint", "fr_steer_right_joint"],
            stiffness=500.0,
            damping=50.0,
            effort_limit_sim=50.0,
        ),
        # ── 전륜 자유회전: 실제 베어링 수준 저마찰 ───────────────────────────
        # damping=0.01: 크루즈(9.695 rad/s)에서 전륜 제동력 1.4N (기존 0.5→70.5N)
        # 구동력 여유: 83.8N→154.2N 대비 저항 70.5N→1.4N → 진동 해소
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=["fr_left_joint", "fr_right_joint"],
            stiffness=0.0,
            damping=0.01,
        ),
    },
)
