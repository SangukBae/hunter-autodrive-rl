# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MuSHR Nano 키보드 텔레오퍼레이션 스크립트 (UWPRL / mushr_nano.usd).

WheeledLab MuSHR Nano 로봇을 키보드로 실시간 조종합니다.
4WD + Ackermann 전륜 조향 구조.

관절 구조:
    조향 (position 제어, ImplicitActuator):
        front_left_wheel_steer   — 좌 전륜 조향
        front_right_wheel_steer  — 우 전륜 조향

    구동 (velocity 제어, DCMotor):
        back_left_wheel_throttle
        back_right_wheel_throttle
        front_left_wheel_throttle
        front_right_wheel_throttle

키 바인딩:
    ─────────────────────────────────────────────
    W / ↑       전진   (누르는 동안 목표 속도 유지)
    S / ↓       후진   (누르는 동안 목표 속도 유지)
    A / ←       좌회전 (누르는 동안 목표 조향각 유지)
    D / →       우회전 (누르는 동안 목표 조향각 유지)
    SPACE       즉시 정지 (속도 + 조향 모두 0)
    1           저속 모드  (0.5 m/s)
    2           중속 모드  (1.5 m/s) [기본값]
    3           고속 모드  (3.0 m/s)
    R           로봇 위치 초기화
    ESC / Q     종료
    ─────────────────────────────────────────────

사용 예시:
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_mushr.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MuSHR Nano 키보드 텔레오퍼레이션")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import math
import weakref

import carb
import carb.input
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat

import sys
sys.path.insert(0, "/robot_isaac/ros2_ws/src/WheeledLab/source/wheeledlab_assets")
from wheeledlab_assets.mushr import MUSHR_CFG

# ── 시뮬레이션 상수 ────────────────────────────────────────────────────────────
DT              = 1 / 200
RENDER_INTERVAL = 4
STEP_DT         = DT * RENDER_INTERVAL   # 0.02 s
HUD_INTERVAL    = 10

# ── MuSHR 파라미터 ─────────────────────────────────────────────────────────────
# MuSHR Nano 1/10 스케일, 바퀴 반지름 약 57mm
WHEEL_RADIUS    = 0.057   # m (추정값; USD 실측값으로 조정 가능)
MAX_STEER       = 0.5     # rad (~28°)
MAX_STEER_RATE  = 1.0     # rad/s

SPEED_LEVELS = {
    "1": 0.5,
    "2": 1.5,
    "3": 3.0,
}
DEFAULT_SPEED_KEY = "2"

# 관절 이름
STEER_LEFT_NAME  = "front_left_wheel_steer"
STEER_RIGHT_NAME = "front_right_wheel_steer"
THROTTLE_NAMES   = [
    "back_left_wheel_throttle",
    "back_right_wheel_throttle",
    "front_left_wheel_throttle",
    "front_right_wheel_throttle",
]

SPAWN_HEIGHT = 0.1   # m


# ── 키보드 컨트롤러 ────────────────────────────────────────────────────────────

class KeyboardController:
    """carb.input 기반 Ackermann 키보드 컨트롤러."""

    def __init__(self, max_speed: float = 1.5, max_steer: float = MAX_STEER):
        self._max_speed  = max_speed
        self._max_steer  = max_steer
        self._reset_flag = False
        self._quit_flag  = False
        self._keys: set[str] = set()

        self._appwindow    = omni.appwindow.get_default_app_window()
        self._input        = carb.input.acquire_input_interface()
        self._keyboard     = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )
        self._print_help()

    def __del__(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)

    @property
    def reset_requested(self) -> bool:
        if self._reset_flag:
            self._reset_flag = False
            return True
        return False

    @property
    def quit_requested(self) -> bool:
        return self._quit_flag

    def get_command(self) -> tuple[float, float]:
        """현재 홀드 상태에서 (lin_vel [m/s], steer_angle [rad]) 반환."""
        if "SPACE" in self._keys:
            return 0.0, 0.0

        lin_vel = 0.0
        if "W" in self._keys or "UP" in self._keys:
            lin_vel = +self._max_speed
        if "S" in self._keys or "DOWN" in self._keys:
            lin_vel = -self._max_speed

        steer = 0.0
        if "A" in self._keys or "LEFT" in self._keys:
            steer = +self._max_steer
        if "D" in self._keys or "RIGHT" in self._keys:
            steer = -self._max_steer

        return lin_vel, steer

    def _on_event(self, event, *args, **kwargs):
        raw      = event.input
        key_name = raw if isinstance(raw, str) else raw.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys.add(key_name)

            if key_name in SPEED_LEVELS:
                self._max_speed = SPEED_LEVELS[key_name]
                print(f"\n[텔레오프] 속도 모드: {self._max_speed:.1f} m/s")
            elif key_name == "R":
                self._reset_flag = True
                print("\n[텔레오프] 위치 초기화")
            elif key_name in ("ESCAPE", "Q"):
                self._quit_flag = True
                print("\n[텔레오프] 종료 요청")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys.discard(key_name)

        return True

    def _print_help(self):
        print("\n" + "=" * 54)
        print("  MuSHR Nano 키보드 텔레오퍼레이션")
        print("=" * 54)
        print("  W / ↑   전진       S / ↓   후진")
        print("  A / ←   좌회전    D / →   우회전")
        print("  SPACE   즉시 정지")
        print("  1       저속 0.5 m/s")
        print("  2       중속 1.5 m/s  [기본값]")
        print("  3       고속 3.0 m/s")
        print("  R       위치 초기화")
        print("  Q / ESC 종료")
        print("=" * 54 + "\n")


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

    robot_cfg = MUSHR_CFG.replace(
        prim_path="/World/MuSHR",
        init_state=MUSHR_CFG.init_state.replace(pos=(0.0, 0.0, SPAWN_HEIGHT)),
    )
    return Articulation(robot_cfg)


# ── 로봇 위치 초기화 ──────────────────────────────────────────────────────────

def reset_robot(robot: Articulation, sim: SimulationContext) -> None:
    dev   = robot.device
    zeros = torch.zeros(1, device=dev)

    root_state = robot.data.default_root_state.clone()
    root_state[0, :3] = torch.tensor([0.0, 0.0, SPAWN_HEIGHT], device=dev)
    root_state[0, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)[0]
    root_state[0, 7:]  = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        robot.data.default_joint_vel,
    )
    sim.step()
    robot.update(STEP_DT)


# ── HUD 출력 ──────────────────────────────────────────────────────────────────

def print_hud(
    step: int,
    robot: Articulation,
    idx_steer_l: int, idx_steer_r: int,
    v_cmd: float, steer_cmd: float,
    omega_rad: float,
) -> None:
    pos   = robot.data.root_pos_w[0]
    vel   = robot.data.root_lin_vel_b[0]
    speed = float(vel.norm())

    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
    yaw_deg   = math.degrees(float(yaw[0]))

    jpos   = robot.data.joint_pos[0]
    sl_act = math.degrees(float(jpos[idx_steer_l]))
    sr_act = math.degrees(float(jpos[idx_steer_r]))

    t = step * STEP_DT
    print(
        f"\r[{t:6.1f}s] "
        f"pos=({pos[0].item():6.2f},{pos[1].item():6.2f}) "
        f"yaw={yaw_deg:6.1f}° spd={speed:.2f}m/s | "
        f"cmd v={v_cmd:+.2f}m/s δ={math.degrees(steer_cmd):+5.1f}° "
        f"ω={omega_rad:+6.1f}rad/s | "
        f"조향 L={sl_act:+5.1f}° R={sr_act:+5.1f}°  ",
        end="",
        flush=True,
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
    sim.set_camera_view(eye=[2.0, -2.0, 1.5], target=[0.0, 0.0, 0.0])

    robot = design_scene()
    sim.reset()

    jnames       = robot.joint_names
    idx_steer_l  = jnames.index(STEER_LEFT_NAME)
    idx_steer_r  = jnames.index(STEER_RIGHT_NAME)
    idx_throttles = [jnames.index(n) for n in THROTTLE_NAMES]

    controller = KeyboardController(
        max_speed=SPEED_LEVELS[DEFAULT_SPEED_KEY],
        max_steer=MAX_STEER,
    )

    prev_steer   = 0.0
    steer_smooth = 0.0
    v_cmd        = 0.0
    omega_rad    = 0.0
    step         = 0

    print("[텔레오프] 시뮬레이션 시작. 뷰포트 창에 포커스 후 키보드 입력.")
    print(f"[텔레오프] 관절 수: {len(jnames)}, 조향: {STEER_LEFT_NAME}/{STEER_RIGHT_NAME}")

    while simulation_app.is_running():

        if controller.quit_requested:
            break

        if controller.reset_requested:
            reset_robot(robot, sim)
            prev_steer   = 0.0
            steer_smooth = 0.0
            v_cmd        = 0.0
            omega_rad    = 0.0
            step         = 0
            continue

        v_target, steer_target = controller.get_command()

        # 조향 rate limiting
        max_change   = MAX_STEER_RATE * STEP_DT
        steer_smooth = prev_steer + max(
            -max_change, min(max_change, steer_target - prev_steer)
        )
        prev_steer = steer_smooth
        v_cmd      = v_target

        omega_rad = v_cmd / WHEEL_RADIUS

        dev = robot.device

        # 조향 적용 (좌우 같은 각도 — 단순 텔레오프)
        robot.set_joint_position_target(
            torch.tensor([[steer_smooth, steer_smooth]], device=dev),
            joint_ids=[idx_steer_l, idx_steer_r],
        )

        # 구동 적용 (4WD 동일 속도)
        # 참고: USD 관절 축 방향에 따라 좌/우 중 한쪽 부호가 반대일 수 있음
        #       로봇이 제자리에서 회전하면 right 관절 (인덱스 1, 3) 부호 반전
        n_throttle = len(idx_throttles)
        robot.set_joint_velocity_target(
            torch.full((1, n_throttle), omega_rad, device=dev),
            joint_ids=idx_throttles,
        )

        robot.write_data_to_sim()
        sim.step()
        robot.update(STEP_DT)
        step += 1

        if step % HUD_INTERVAL == 0:
            print_hud(
                step, robot,
                idx_steer_l, idx_steer_r,
                v_cmd, steer_smooth, omega_rad,
            )

    print("\n[텔레오프] 종료.")


if __name__ == "__main__":
    main()
    simulation_app.close()
