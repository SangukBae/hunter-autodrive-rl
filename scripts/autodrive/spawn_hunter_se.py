# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE (URDF→USD 변환본) 스폰 테스트 스크립트.

Isaac Sim 뷰포트에서 Hunter SE 로봇을 스폰하고 조인트 정보를 출력합니다.

사용 예시:
    # GUI 모드 (로봇 확인)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/spawn_hunter_se.py

    # 헤드리스 모드 (조인트 정보만 출력)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/spawn_hunter_se.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE 스폰 테스트")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Hunter SE ArticulationCfg (변환본)
import sys, os
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se.hunter_se_cfg import HUNTER_SE_CFG


def design_scene() -> Articulation:
    """씬 구성: 평지 + 조명 + Hunter SE 로봇."""

    # 평지
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    # 조명
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Hunter SE 스폰
    # 스폰 테스트에서는 contact sensor 불필요 → activate_contact_sensors=False로 오버라이드
    # (GPU Direct API 활성화 상태에서 contact sensor 초기화가 CPU-side addForce를 호출하는 충돌 방지)
    robot_cfg = HUNTER_SE_CFG.replace(
        prim_path="/World/HunterSE",
        spawn=HUNTER_SE_CFG.spawn.replace(activate_contact_sensors=False),
    )
    robot = Articulation(robot_cfg)

    return robot


def print_robot_info(robot: Articulation) -> None:
    """로봇 조인트/링크 정보 출력."""
    print("\n" + "=" * 60)
    print("Hunter SE 로봇 정보")
    print("=" * 60)
    print(f"  조인트 수     : {robot.num_joints}")
    print(f"  바디 수       : {robot.num_bodies}")
    print(f"  조인트 이름   :")
    for i, name in enumerate(robot.joint_names):
        print(f"    [{i:2d}] {name}")
    print(f"  바디 이름     :")
    for i, name in enumerate(robot.body_names):
        print(f"    [{i:2d}] {name}")
    print("=" * 60 + "\n")


def main():
    """메인 루프."""

    # 시뮬레이션 설정
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.3])

    # 씬 구성
    robot = design_scene()

    # 시뮬레이션 리셋 (물리 초기화)
    sim.reset()

    # 로봇 정보 출력
    print_robot_info(robot)

    # 후륜 조인트 인덱스 탐색
    try:
        lw_idx, _ = robot.find_joints("re_left_joint")
        rw_idx, _ = robot.find_joints("re_right_joint")
        fsl_idx, _ = robot.find_joints("fr_steer_left_joint")
        fsr_idx, _ = robot.find_joints("fr_steer_right_joint")
        print(f"[INFO] 후륜 좌  인덱스: {lw_idx}")
        print(f"[INFO] 후륜 우  인덱스: {rw_idx}")
        print(f"[INFO] 조향 좌  인덱스: {fsl_idx}")
        print(f"[INFO] 조향 우  인덱스: {fsr_idx}")
    except Exception as e:
        print(f"[WARN] 조인트 탐색 실패: {e}")

    # 시뮬레이션 루프 (GUI 모드에서 뷰포트 확인 가능)
    step = 0
    while simulation_app.is_running():
        # 물리 스텝
        sim.step()
        robot.update(sim_cfg.dt)
        step += 1

        # 5초마다 속도/위치 출력
        if step % (200 * 5) == 0:
            pos = robot.data.root_pos_w[0]
            vel = robot.data.root_lin_vel_b[0]
            print(f"[Step {step:6d}] pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                  f"vel_x={vel[0]:.3f} m/s")

        if args_cli.headless and step >= 200:
            print("[INFO] 헤드리스 모드: 200 스텝 후 종료")
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
