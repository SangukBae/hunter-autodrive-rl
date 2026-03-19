# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE AutoDRIVE Ackermann 주행 테스트 스크립트.

/robot_isaac/ros2_ws/src/hunter_se_autodrive/ 에서
/workspace/hunter_autodrive/hunter_se_autodrive/ 로 이동한
AutoDRIVE 버전 로봇으로 주행 테스트를 수행합니다.

Hunter SE AutoDRIVE 제원 (HunterSESceneBuilder.cs 기준, 1:5 스케일):
    - 축거         : 0.550 m
    - 윤거 (전/후) : 0.520 m
    - 바퀴 반경    : 0.082 m
    - 최대 조향각  : ±30° = ±0.5236 rad
    - 최고 속도    : 3.5611 m/s

주행 시나리오:
    1. 직진     5초  (v=1.0 m/s)
    2. 좌회전   5초  (v=1.0 m/s, δ=0.4 rad ≈ 22.9°)
    3. 직진     5초  (v=1.0 m/s)
    4. 우회전   5초  (v=1.0 m/s, δ=0.4 rad)
    5. 감속 정지 3초

검증 항목:
    - 직진 시 X축 방향 이동 (Y 드리프트 최소)
    - 회전 시 원호 궤적 (위치 변화 패턴)
    - 후륜 구동 + 전륜 조향 올바른 동작
    - 속도 응답 (1.0 m/s 목표 추종)

사용 예시:
    # GUI 모드 (뷰포트에서 궤적 확인 권장)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_hunter_se_autodrive.py

    # 헤드리스 모드 (수치 출력만)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_hunter_se_autodrive.py --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE AutoDRIVE Ackermann 주행 테스트")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

import sys
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_autodrive.hunter_se_autodrive_cfg import HUNTER_SE_AUTODRIVE_CFG

# ── Hunter SE AutoDRIVE 물리 상수 ─────────────────────────────────────────────
DT = 1 / 200          # 물리 스텝 주기 [s]
RENDER_INTERVAL = 4
STEP_DT = DT * RENDER_INTERVAL  # sim.step() 1회당 실제 경과 시간 [s] (= 0.02s)

WHEELBASE    = 0.550   # 축거 [m]
FRONT_TRACK  = 0.520   # 전륜 조향 피벗 간격 [m]
REAR_TRACK   = 0.520   # 후륜 허브 간격 [m]
WHEEL_RADIUS = 0.082   # 바퀴 반지름 [m]
MAX_STEER    = 0.5236  # 최대 내륜 조향각 [rad] (±30°)
MAX_SPEED    = 3.5611  # 최대 선속도 [m/s]

# 관절 이름으로 인덱스를 동적으로 찾기 위한 이름 목록
STEER_LEFT_NAME  = "fr_steer_left_joint"
STEER_RIGHT_NAME = "fr_steer_right_joint"
WHEEL_LEFT_NAME  = "re_left_joint"
WHEEL_RIGHT_NAME = "re_right_joint"


# ── Ackermann 조향 계산 ────────────────────────────────────────────────────────

def ackermann_cmd(v: float, delta: float) -> tuple:
    """Ackermann 조향 기하학으로 관절 명령 계산.

    Args:
        v:     차량 선속도 [m/s] (양수=전진, 음수=후진)
        delta: 내륜 조향각 [rad] (양수=좌회전, 음수=우회전)

    Returns:
        (steer_left, steer_right, omega_left, omega_right)
            steer_*: 조향 관절 위치 목표값 [rad]
            omega_*: 후륜 관절 속도 목표값 [rad/s]
    """
    v = max(-MAX_SPEED, min(MAX_SPEED, v))
    delta = max(-MAX_STEER, min(MAX_STEER, delta))

    # 직진
    if abs(delta) < 1e-4:
        omega = v / WHEEL_RADIUS
        return 0.0, 0.0, omega, omega

    # 회전 중심까지 거리 (후륜 축 기준)
    R = WHEELBASE / math.tan(abs(delta))

    # Ackermann: 외륜 조향각
    delta_outer = math.atan(WHEELBASE / (R + FRONT_TRACK))

    # 후륜 차동 각속도
    R_center = R
    omega_inner = v * (R - REAR_TRACK / 2) / (R_center * WHEEL_RADIUS)
    omega_outer = v * (R + REAR_TRACK / 2) / (R_center * WHEEL_RADIUS)

    if delta > 0:
        # 좌회전: 좌 = 내륜(큰 각도, 느린 바퀴), 우 = 외륜
        return +abs(delta), +delta_outer, omega_inner, omega_outer
    else:
        # 우회전: 우 = 내륜, 좌 = 외륜
        return -delta_outer, -abs(delta), omega_outer, omega_inner


# ── 씬 구성 ────────────────────────────────────────────────────────────────────

def design_scene() -> Articulation:
    """평지 + 조명 + Hunter SE AutoDRIVE 스폰."""
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

    robot_cfg = HUNTER_SE_AUTODRIVE_CFG.replace(
        prim_path="/World/HunterSEAutoDRIVE",
        spawn=HUNTER_SE_AUTODRIVE_CFG.spawn.replace(activate_contact_sensors=False),
    )
    return Articulation(robot_cfg)


# ── 제어 명령 적용 ─────────────────────────────────────────────────────────────

def apply_action(robot: Articulation,
                 idx_steer_l: int, idx_steer_r: int,
                 idx_wheel_l: int, idx_wheel_r: int,
                 steer_l: float, steer_r: float,
                 omega_l: float, omega_r: float) -> None:
    """조향각 + 후륜 속도 명령 한 번에 적용."""
    dev = robot.device
    robot.set_joint_position_target(
        torch.tensor([[steer_l, steer_r]], device=dev),
        joint_ids=[idx_steer_l, idx_steer_r],
    )
    robot.set_joint_velocity_target(
        torch.tensor([[omega_l, omega_r]], device=dev),
        joint_ids=[idx_wheel_l, idx_wheel_r],
    )
    robot.write_data_to_sim()


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
    sim.set_camera_view(eye=[4.0, -4.0, 3.0], target=[2.0, 0.0, 0.0])

    robot = design_scene()
    sim.reset()

    # 관절 인덱스 동적 탐색
    jnames = robot.joint_names
    idx_steer_l = jnames.index(STEER_LEFT_NAME)
    idx_steer_r = jnames.index(STEER_RIGHT_NAME)
    idx_wheel_l = jnames.index(WHEEL_LEFT_NAME)
    idx_wheel_r = jnames.index(WHEEL_RIGHT_NAME)

    print("\n" + "=" * 70)
    print("Hunter SE AutoDRIVE 주행 테스트")
    for i, name_j in enumerate(jnames):
        print(f"  [{i}] {name_j}")
    print(f"  WHEELBASE={WHEELBASE}m  REAR_TRACK={REAR_TRACK}m  "
          f"WHEEL_RADIUS={WHEEL_RADIUS}m")
    print(f"  IDX_STEER_LEFT={idx_steer_l}→{jnames[idx_steer_l]}  "
          f"IDX_STEER_RIGHT={idx_steer_r}→{jnames[idx_steer_r]}")
    print(f"  IDX_WHEEL_LEFT={idx_wheel_l}→{jnames[idx_wheel_l]}  "
          f"IDX_WHEEL_RIGHT={idx_wheel_r}→{jnames[idx_wheel_r]}")
    print("=" * 70)

    # ── PhysX 실제 설정값 진단 ─────────────────────────────────────────────────
    import numpy as np
    stiff = robot.data.joint_stiffness[0].cpu().numpy()
    damp  = robot.data.joint_damping[0].cpu().numpy()
    elim  = robot.data.joint_effort_limits[0].cpu().numpy()
    print("\n[PhysX 설정 진단]")
    print(f"  {'관절':<28} {'stiffness':>12} {'damping':>12} {'effort_lim':>12}")
    for i, n in enumerate(jnames):
        print(f"  [{i}] {n:<24} {stiff[i]:>12.1f} {damp[i]:>12.1f} {elim[i]:>12.3f}")
    print()

    # ── 주행 시나리오 정의 ──────────────────────────────────────────────────────
    # (설명, 선속도[m/s], 내륜조향각[rad], 지속시간[s])
    SCENARIOS = [
        ("직진",   1.0,  0.000, 5.0),
        ("좌회전", 1.0, +0.400, 5.0),
        ("직진",   1.0,  0.000, 5.0),
        ("우회전", 1.0, -0.400, 5.0),
        ("정지",   0.0,  0.000, 3.0),
    ]

    phase_start_pos = None
    prev_pos = None

    print(f"\n{'시간':>6}  {'단계':^6}  {'X[m]':>7}  {'Y[m]':>7}  {'Z[m]':>7}  "
          f"{'속도[m/s]':>9}  {'조향L':>7}  {'조향R':>7}  {'이동거리':>8}")
    print("-" * 70)

    step = 0
    sc_idx = 0
    sc_step = 0
    name, v_cmd, delta_cmd, duration = SCENARIOS[sc_idx]
    sc_total = int(duration / STEP_DT)
    print_interval = int(1.0 / STEP_DT)   # 1초마다 출력

    while simulation_app.is_running():

        # 명령 계산 및 적용
        sl, sr, wl, wr = ackermann_cmd(v_cmd, delta_cmd)
        apply_action(robot,
                     idx_steer_l, idx_steer_r,
                     idx_wheel_l, idx_wheel_r,
                     sl, sr, wl, wr)

        sim.step()
        robot.update(STEP_DT)
        step += 1
        sc_step += 1

        # 1초마다 상태 출력
        if step % print_interval == 0:
            pos = robot.data.root_pos_w[0]
            vel = robot.data.root_lin_vel_b[0]
            speed = float(vel.norm())

            dist = 0.0
            if prev_pos is not None:
                diff = pos - prev_pos
                dist = float(diff.norm())
            prev_pos = pos.clone()

            w_vel = robot.data.joint_vel[0]
            w_pos = robot.data.joint_pos[0]
            wl_act = float(w_vel[idx_wheel_l])
            wr_act = float(w_vel[idx_wheel_r])
            sl_act = float(w_pos[idx_steer_l])
            sr_act = float(w_pos[idx_steer_r])

            print(f"{step * STEP_DT:6.1f}s  {name:^6}  "
                  f"{pos[0].item():7.3f}  {pos[1].item():7.3f}  {pos[2].item():7.3f}  "
                  f"{speed:9.4f}  "
                  f"{math.degrees(sl):7.2f}°  {math.degrees(sr):7.2f}°  "
                  f"{dist:7.4f}m")
            print(f"         [진단] 후륜ω: L={wl_act:6.3f} R={wr_act:6.3f} rad/s (목표:{wl:.2f})  "
                  f"조향실제: L={math.degrees(sl_act):6.2f}° R={math.degrees(sr_act):6.2f}°")

        # 단계 전환
        if sc_step >= sc_total:
            cur_pos = robot.data.root_pos_w[0]
            if phase_start_pos is not None:
                dx = float(cur_pos[0] - phase_start_pos[0])
                dy = float(cur_pos[1] - phase_start_pos[1])
                seg_dist = math.sqrt(dx * dx + dy * dy)
                print(f"  ▶ [{name}] 구간 이동: Δx={dx:+.3f}m  Δy={dy:+.3f}m  "
                      f"수평거리={seg_dist:.3f}m")

            sc_idx += 1
            sc_step = 0
            if sc_idx >= len(SCENARIOS):
                print("\n" + "=" * 70)
                print("[INFO] 모든 주행 시나리오 완료")
                print("=" * 70)
                break

            name, v_cmd, delta_cmd, duration = SCENARIOS[sc_idx]
            sc_total = int(duration / STEP_DT)
            phase_start_pos = robot.data.root_pos_w[0].clone()
            prev_pos = None
            print(f"\n  ── [{name}]  v={v_cmd:.1f} m/s  "
                  f"δ={math.degrees(delta_cmd):.1f}° ──")


if __name__ == "__main__":
    main()
    simulation_app.close()
