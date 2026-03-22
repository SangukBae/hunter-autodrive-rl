# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 최고속도 직진 검증 스크립트.

공식 메뉴얼 최고속도 1.33 m/s 로 hunter_se_v0 로봇을 직진시키고
터미널에 실시간 주행 데이터를 출력합니다. 키보드 입력 없이 자동 실행.

사용 예시:
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_straight_hunter_se_v0.py
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_straight_hunter_se_v0.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE V0 최고속도 직진 검증")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import sys
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_v0.hunter_se_v0_cfg import HUNTER_SE_V0_CFG
from hunter_se_v0.ackermann import HunterSEAckermann, MAX_SPEED

# ── 시뮬레이션 상수 ────────────────────────────────────────────────────────────
DT              = 1 / 200
RENDER_INTERVAL = 4
STEP_DT         = DT * RENDER_INTERVAL    # 0.02 s

# 목표 속도: 공식 메뉴얼 최고속도
TARGET_SPEED = MAX_SPEED   # 1.333 m/s
TARGET_DELTA = 0.0         # 직진 (조향각 0)

# 관절 이름
STEER_LEFT_NAME  = "fr_steer_left_joint"
STEER_RIGHT_NAME = "fr_steer_right_joint"
WHEEL_LEFT_NAME  = "re_left_joint"
WHEEL_RIGHT_NAME = "re_right_joint"

SPAWN_Z      = 0.2955   # WHEEL_RADIUS(0.1375) + |REAR_AX_Z|(0.158)
HUD_INTERVAL = 10       # HUD 출력 간격 (스텝 수)


# ── 씬 구성 ────────────────────────────────────────────────────────────────────

def design_scene() -> Articulation:
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = HUNTER_SE_V0_CFG.replace(
        prim_path="/World/HunterSEV0",
        spawn=HUNTER_SE_V0_CFG.spawn.replace(activate_contact_sensors=False),
    )
    return Articulation(robot_cfg)


# ── HUD 출력 ──────────────────────────────────────────────────────────────────

def print_hud(
    step: int,
    robot: Articulation,
    idx_steer_l: int, idx_steer_r: int,
    idx_wheel_l: int,
    omega_cmd_l: float,
) -> None:
    vel         = robot.data.root_lin_vel_b[0]
    speed       = float(vel[0])           # X축 전진 속도 (부호 포함)
    speed_abs   = float(vel.norm())
    wheel_l_vel = float(robot.data.joint_vel[0][idx_wheel_l])

    t = step * STEP_DT
    print(
        f"[{t:7.2f}s] "
        f"속력={speed_abs:.4f}m/s (Vx={speed:+.4f}) "
        f"cmd={TARGET_SPEED:.3f}m/s | "
        f"후륜좌 ω={wheel_l_vel:+7.4f}rad/s  cmdω={omega_cmd_l:+7.4f}rad/s"
    )


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=DT,
        render_interval=RENDER_INTERVAL,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -4.0, 2.0], target=[0.0, 0.0, 0.0])

    robot = design_scene()
    sim.reset()

    jnames      = robot.joint_names
    idx_steer_l = jnames.index(STEER_LEFT_NAME)
    idx_steer_r = jnames.index(STEER_RIGHT_NAME)
    idx_wheel_l = jnames.index(WHEEL_LEFT_NAME)
    idx_wheel_r = jnames.index(WHEEL_RIGHT_NAME)

    acker = HunterSEAckermann(device=robot.device)

    # 직진 Ackermann 값 사전 계산 (변하지 않음)
    steer_l, steer_r, omega_l, omega_r = acker.compute_scalar(TARGET_SPEED, TARGET_DELTA)

    print("\n" + "=" * 60)
    print(f"  Hunter SE V0 직진 검증  —  목표속도 {TARGET_SPEED:.3f} m/s")
    print("=" * 60)
    print(f"  Ackermann 사전 계산:")
    print(f"    steer_L = {math.degrees(steer_l):+.4f}°  steer_R = {math.degrees(steer_r):+.4f}°")
    print(f"    ω_L     = {omega_l:+.4f} rad/s  ω_R = {omega_r:+.4f} rad/s")
    print(f"  종료: Ctrl+C")
    print("=" * 60 + "\n")

    dev  = robot.device
    step = 0

    steer_tensor = torch.tensor([[steer_l, steer_r]], device=dev)
    omega_tensor = torch.tensor([[omega_l, omega_r]], device=dev)

    while simulation_app.is_running():
        robot.set_joint_position_target(steer_tensor, joint_ids=[idx_steer_l, idx_steer_r])
        robot.set_joint_velocity_target(omega_tensor, joint_ids=[idx_wheel_l, idx_wheel_r])
        robot.write_data_to_sim()

        sim.step()
        robot.update(STEP_DT)
        step += 1

        if step % HUD_INTERVAL == 0:
            print_hud(step, robot, idx_steer_l, idx_steer_r, idx_wheel_l, omega_l)


if __name__ == "__main__":
    main()
    simulation_app.close()
