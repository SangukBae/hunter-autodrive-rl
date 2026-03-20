# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 키보드 텔레오퍼레이션 스크립트.

기본 도형(Box/Cylinder) 기반 hunter_se_v0 로봇을 Isaac Sim에 스폰하고
키보드로 실시간 조종합니다.

키 바인딩:
    ─────────────────────────────────────────────
    W / ↑       전진   (누르는 동안 목표 속도 유지)
    S / ↓       후진   (누르는 동안 목표 속도 유지)
    A / ←       좌회전 (누르는 동안 목표 조향각 유지)
    D / →       우회전 (누르는 동안 목표 조향각 유지)
    SPACE       즉시 정지 (속도 + 조향 모두 0)
    1           저속 모드  (0.3 m/s)
    2           중속 모드  (0.7 m/s)  [기본값]
    3           고속 모드  (1.3 m/s)
    R           로봇 위치 초기화
    ESC / Q     종료
    ─────────────────────────────────────────────

사용 예시:
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_hunter_se_v0.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE V0 키보드 텔레오퍼레이션")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False   # GUI 모드 강제 (키보드 입력에 창 필요)

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
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import sys
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_v0.hunter_se_v0_cfg import HUNTER_SE_V0_CFG
from hunter_se_v0.ackermann import HunterSEAckermann, MAX_STEER, MAX_SPEED

# ── 시뮬레이션 상수 ────────────────────────────────────────────────────────────
DT              = 1 / 200
RENDER_INTERVAL = 4
STEP_DT         = DT * RENDER_INTERVAL    # 0.02 s

# 조향 변화율 제한 (rad/s) — RL 환경과 동일
MAX_STEER_RATE  = 0.5

# 속도 단계 테이블
SPEED_LEVELS = {
    "KEY_1": 0.3,
    "KEY_2": 0.7,
    "KEY_3": MAX_SPEED,
}
DEFAULT_SPEED_KEY = "KEY_2"

# 관절 이름
STEER_LEFT_NAME  = "fr_steer_left_joint"
STEER_RIGHT_NAME = "fr_steer_right_joint"
WHEEL_LEFT_NAME  = "re_left_joint"
WHEEL_RIGHT_NAME = "re_right_joint"

HUD_INTERVAL = 10   # HUD 출력 간격 (스텝 수)


# ── 키보드 컨트롤러 ────────────────────────────────────────────────────────────

class HunterKeyboardController:
    """carb.input 기반 키보드 컨트롤러.

    방향키/WASD 홀드로 목표 속도·조향각을 유지합니다.
    """

    def __init__(self, max_speed: float = 0.7):
        self._max_speed  = max_speed
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
        """현재 홀드 키 기준 목표 (lin_vel [m/s], delta_c [rad]) 반환."""
        if "SPACE" in self._keys:
            return 0.0, 0.0

        lin_vel = 0.0
        if "W" in self._keys or "UP" in self._keys:
            lin_vel = +self._max_speed
        if "S" in self._keys or "DOWN" in self._keys:
            lin_vel = -self._max_speed

        delta_c = 0.0
        if "A" in self._keys or "LEFT" in self._keys:
            delta_c = +MAX_STEER
        if "D" in self._keys or "RIGHT" in self._keys:
            delta_c = -MAX_STEER

        return lin_vel, delta_c

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
                print("\n[텔레오프] 로봇 위치 초기화")
            elif key_name in ("ESCAPE", "Q"):
                self._quit_flag = True
                print("\n[텔레오프] 종료 요청")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys.discard(key_name)

        return True

    def _print_help(self):
        print("\n" + "=" * 56)
        print("  Hunter SE V0 키보드 텔레오퍼레이션")
        print("=" * 56)
        print("  W / ↑   전진       S / ↓   후진")
        print("  A / ←   좌회전    D / →   우회전")
        print("  SPACE   즉시 정지")
        print("  1  저속 0.3 m/s  |  2  중속 0.7 m/s  |  3  고속 1.3 m/s")
        print("  R  위치 초기화   |  Q / ESC  종료")
        print("=" * 56 + "\n")


# ── 씬 구성 ────────────────────────────────────────────────────────────────────

def design_scene() -> Articulation:
    """평지 + 조명 + Hunter SE V0 스폰."""
    # 평지 (마찰 계수: 실제 아스팔트 수준)
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    # 돔 조명
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Hunter SE V0 스폰 (contact sensor 불필요)
    robot_cfg = HUNTER_SE_V0_CFG.replace(
        prim_path="/World/HunterSEV0",
        spawn=HUNTER_SE_V0_CFG.spawn.replace(activate_contact_sensors=False),
    )
    return Articulation(robot_cfg)


# ── 로봇 위치 초기화 ───────────────────────────────────────────────────────────

def reset_robot(robot: Articulation, sim: SimulationContext) -> None:
    """로봇을 원점으로 되돌리고 속도를 0으로 초기화."""
    dev   = robot.device
    zeros = torch.zeros(1, device=dev)

    root_state = robot.data.default_root_state.clone()
    root_state[0, :3] = torch.tensor([0.0, 0.0, 0.287], device=dev)
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
    idx_wheel_l: int,
    v_cmd: float, delta_cmd: float,
    omega_cmd_l: float,
) -> None:
    pos   = robot.data.root_pos_w[0]
    vel   = robot.data.root_lin_vel_b[0]
    speed = float(vel.norm())

    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
    yaw_deg   = math.degrees(float(yaw[0]))

    w_pos       = robot.data.joint_pos[0]
    w_vel       = robot.data.joint_vel[0]
    sl_act      = math.degrees(float(w_pos[idx_steer_l]))
    sr_act      = math.degrees(float(w_pos[idx_steer_r]))
    wheel_l_pos = float(w_pos[idx_wheel_l])
    wheel_l_vel = float(w_vel[idx_wheel_l])

    t = step * STEP_DT
    print(
        f"\r[{t:6.1f}s] "
        f"pos=({pos[0].item():6.2f},{pos[1].item():6.2f}) "
        f"yaw={yaw_deg:6.1f}° "
        f"spd={speed:.3f}m/s | "
        f"cmd v={v_cmd:+.2f} δ={math.degrees(delta_cmd):+5.1f}° | "
        f"조향 L={sl_act:+5.1f}° R={sr_act:+5.1f}° | "
        f"후륜좌 θ={wheel_l_pos:+6.3f}rad "
        f"ω={wheel_l_vel:+6.3f}rad/s "
        f"cmdω={omega_cmd_l:+6.3f}rad/s  ",
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
    sim.set_camera_view(eye=[3.0, -3.0, 2.5], target=[0.0, 0.0, 0.0])

    robot = design_scene()
    sim.reset()

    # 관절 인덱스 탐색
    jnames      = robot.joint_names
    idx_steer_l = jnames.index(STEER_LEFT_NAME)
    idx_steer_r = jnames.index(STEER_RIGHT_NAME)
    idx_wheel_l = jnames.index(WHEEL_LEFT_NAME)
    idx_wheel_r = jnames.index(WHEEL_RIGHT_NAME)

    print(f"\n[INFO] 관절 목록 ({len(jnames)}개):")
    for i, n in enumerate(jnames):
        print(f"  [{i:2d}] {n}")

    # Ackermann 처리기 & 키보드 컨트롤러
    acker      = HunterSEAckermann(device=robot.device)
    controller = HunterKeyboardController(max_speed=SPEED_LEVELS[DEFAULT_SPEED_KEY])

    prev_delta   = 0.0
    v_cmd        = 0.0
    delta_smooth = 0.0
    step         = 0

    print("\n[텔레오프] 시뮬레이션 시작. 뷰포트 창에 포커스 후 키보드 입력.\n")

    while simulation_app.is_running():

        if controller.quit_requested:
            break

        if controller.reset_requested:
            reset_robot(robot, sim)
            prev_delta = delta_smooth = v_cmd = 0.0
            step = 0
            continue

        # 키 명령 읽기
        v_target, delta_target = controller.get_command()

        # 조향 rate limiting
        max_change   = MAX_STEER_RATE * STEP_DT
        delta_smooth = prev_delta + max(
            -max_change, min(max_change, delta_target - prev_delta)
        )
        prev_delta = delta_smooth
        v_cmd      = v_target

        # Ackermann 계산
        steer_l, steer_r, omega_l, omega_r = acker.compute_scalar(
            v_cmd, delta_smooth
        )

        # 관절 명령 적용
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

        # 물리 스텝
        sim.step()
        robot.update(STEP_DT)
        step += 1

        if step % HUD_INTERVAL == 0:
            print_hud(
                step, robot,
                idx_steer_l, idx_steer_r,
                idx_wheel_l,
                v_cmd, delta_smooth,
                omega_l,
            )

    print("\n[텔레오프] 종료.")


if __name__ == "__main__":
    main()
    simulation_app.close()
