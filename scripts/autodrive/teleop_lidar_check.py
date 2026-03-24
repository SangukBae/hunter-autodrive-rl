# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 RTX LiDAR 동작 확인 스크립트.

씬 구성:
    - Hunter SE V0 (원점 스폰, 전방 +X 방향)
    - 주황색 상자 장애물 (로봇 전방 3 m)
    - 평지 + 돔 조명

센서 구성:
    Isaac Sim 5.0+ 공식 방식으로 OmniLidar prim 을 런타임에 생성한다.
    LidarRtx(config_file_name=...) 가 IsaacSensorCreateRtxLidar 커맨드를
    내부적으로 호출하여 OmniLidar prim 과 render product 를 자동 생성한다.

    참고: Isaac Sim 4.5 이하의 Camera prim 방식(sensorModelPluginName)은
    5.0 에서 deprecated 되었으므로 USD 에 내장하지 않는다.

뷰포트 시각화:
    RtxLidarDebugDrawPointCloudBuffer writer 를 사용한다.
    Buffer 버전은 1회전(360°) 완성 후 포인트를 표시하되 다음 스캔이
    완성될 때까지 이전 포인트를 유지하므로 빈 프레임 없이 연속적인
    점군처럼 보인다 (공식 standalone 예제와 동일한 방식).

콘솔 출력 (매 LIDAR_PRINT_INTERVAL 스텝):
    [LiDAR] hits=XXXX  min=0.00m  max=00.00m  front_min=0.00m
    front_min : 전방 ±15° 부채꼴 내 최소 거리 (장애물 감지 지표)

키 바인딩:
    W / ↑       전진      S / ↓   후진
    A / ←       좌회전    D / →   우회전
    SPACE       즉시 정지
    1/2/3       저속 0.3 / 중속 0.7 / 고속 1.3 m/s
    R           로봇 위치 초기화
    ESC / Q     종료

실행:
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_lidar_check.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE V0 RTX LiDAR 확인")
parser.add_argument(
    "--lidar",
    type=str,
    default="os1-32",
    choices=["os1-32", "os1-64", "os1-128"],
    help="LiDAR 모델 선택 (기본: os1-32)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False   # GUI 모드 강제

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import math
import weakref

import carb
import carb.input
import numpy as np
import omni.appwindow
import torch
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.util.debug_draw import _debug_draw as _dd_module

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import sys
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_v0.hunter_se_v0_cfg import HUNTER_SE_V0_CFG
from hunter_se_v0.ackermann import HunterSEAckermann, MAX_STEER, MAX_SPEED
from hunter_se_v0.lidar_cfg import get_lidar_spec, VALID_LIDAR_MODELS

# ── debug_draw 익스텐션 활성화 (공식 예제와 동일) ─────────────────────────────
enable_extension("isaacsim.util.debug_draw")

# ── 상수 ──────────────────────────────────────────────────────────────────────
DT              = 1 / 200
RENDER_INTERVAL = 4
STEP_DT         = DT * RENDER_INTERVAL    # 0.02 s

MAX_STEER_RATE  = 0.5                     # 조향 변화율 한계 [rad/s]

SPEED_LEVELS      = {"KEY_1": 0.3, "KEY_2": 0.7, "KEY_3": MAX_SPEED}
DEFAULT_SPEED_KEY = "KEY_2"

STEER_LEFT_NAME  = "fr_steer_left_joint"
STEER_RIGHT_NAME = "fr_steer_right_joint"
WHEEL_LEFT_NAME  = "re_left_joint"
WHEEL_RIGHT_NAME = "re_right_joint"

ROBOT_PRIM      = "/World/HunterSEV0"
# OmniLidar prim 생성 경로 — base_link 자식으로 두어 로봇 이동 시 자동 추종
LIDAR_PRIM      = f"{ROBOT_PRIM}/base_link/Lidar"
OBSTACLE_PRIM   = "/World/Obstacle"
OBSTACLE_OFFSET = 3.0   # 전방 거리 [m]

# ── LiDAR 모델 선택 (--lidar CLI 인자로 결정) ──────────────────────────────────
# 지원 모델: os1-32, os1-64, os1-128
# os1-64 주의: Isaac Sim 5.0 에 내장 RTX config 없음 → OS1-128 config 로 대체 실행.
#             MultiMeshRayCaster 학습 환경은 channels=64 를 그대로 사용 가능.
_lidar_spec      = get_lidar_spec(args_cli.lidar)
LIDAR_CONFIG     = _lidar_spec["rtx_config"]   # Isaac Sim RTX config 이름
LIDAR_PTS_PER_SCAN = _lidar_spec["pts_per_scan"]
_lidar_channels  = _lidar_spec["channels"]
_lidar_pps       = _lidar_spec["points_per_second"]

LIDAR_Z         = 0.5   # base_link 기준 LiDAR 마운트 높이 [m]

# 포인트 클라우드 어노테이터 (콘솔 통계 + 시각화 공용)
LIDAR_ANNOT          = "IsaacExtractRTXSensorPointCloudNoAccumulator"
LIDAR_PRINT_INTERVAL = 20   # 콘솔 출력 간격 (스텝 수)
FRONT_HALF_ANGLE_DEG = 15.0 # 전방 부채꼴 반각 [°]

HUD_INTERVAL = 10


# ── 키보드 컨트롤러 ────────────────────────────────────────────────────────────

class HunterKeyboardController:
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
        print("\n" + "=" * 60)
        print("  Hunter SE V0  RTX LiDAR 확인 텔레오퍼레이션")
        print("=" * 60)
        print("  W/↑  전진       S/↓  후진")
        print("  A/←  좌회전    D/→  우회전")
        print("  SPACE  즉시 정지")
        print("  1  저속 0.3 m/s  |  2  중속 0.7 m/s  |  3  고속 1.3 m/s")
        print("  R  위치 초기화  |  Q / ESC  종료")
        print("-" * 60)
        print(f"  LiDAR 모델: {args_cli.lidar}  ({_lidar_channels}ch / 10Hz / PPS={_lidar_pps:,})")
        print("  뷰포트: RTX LiDAR 포인트 클라우드 자동 표시")
        print("  콘솔:  hits / min·max 거리 / front_min 출력")
        print("  front_min ≈ 3.0m → 전진 시 감소 = 장애물 감지 성공")
        print("=" * 60 + "\n")


# ── 씬 구성 ────────────────────────────────────────────────────────────────────

def design_scene() -> Articulation:
    """평지 + 조명 + 로봇 + 장애물 상자 스폰."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 주황색 상자 장애물 (0.5 × 0.5 × 1.0 m), 전방 3 m
    box_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.45, 0.0), metallic=0.0,
        ),
    )
    box_cfg.func(OBSTACLE_PRIM, box_cfg, translation=(OBSTACLE_OFFSET, 0.0, 0.5))
    print(f"[INFO] 장애물 스폰: {OBSTACLE_PRIM}  위치=({OBSTACLE_OFFSET:.1f}, 0, 0.5)")

    robot_cfg = HUNTER_SE_V0_CFG.replace(
        prim_path=ROBOT_PRIM,
        spawn=HUNTER_SE_V0_CFG.spawn.replace(activate_contact_sensors=False),
    )
    robot = Articulation(robot_cfg)
    print(f"[INFO] 로봇 스폰: {ROBOT_PRIM}")
    return robot


# ── LiDAR 통계 계산 ────────────────────────────────────────────────────────────

def _lidar_stats(frame: dict) -> str:
    """get_current_frame() 딕셔너리에서 hits / 거리 통계 문자열을 반환한다.

    IsaacExtractRTXSensorPointCloudNoAccumulator 어노테이터는 현재 렌더
    프레임의 부분 스캔 포인트를 월드 프레임 XYZ float32 (N×3) 로 반환한다.
    """
    raw = frame.get(LIDAR_ANNOT, None)
    if raw is None:
        return "(데이터 없음 — 아직 초기화 중)"

    pts = raw.get("data", None) if isinstance(raw, dict) else raw
    if pts is None or (hasattr(pts, "__len__") and len(pts) == 0):
        return "hits=0"

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return f"hits=? (shape={pts.shape})"

    dist  = np.linalg.norm(pts[:, :3], axis=1)
    valid = (dist > 0.05) & np.isfinite(dist)
    n_hits = int(valid.sum())
    if n_hits == 0:
        return f"hits=0/{len(pts)}"

    vd    = dist[valid]
    min_d = float(vd.min())
    max_d = float(vd.max())

    half_rad  = math.radians(FRONT_HALF_ANGLE_DEG)
    az        = np.arctan2(pts[valid, 1], pts[valid, 0])
    front     = vd[np.abs(az) < half_rad]
    front_str = f"  front_min={float(front.min()):.2f}m" if len(front) > 0 else ""

    return f"hits={n_hits}/{len(pts)}  min={min_d:.2f}m  max={max_d:.2f}m{front_str}"


# ── 로봇 위치 초기화 ───────────────────────────────────────────────────────────

def reset_robot(robot: Articulation, sim: SimulationContext) -> None:
    dev   = robot.device
    zeros = torch.zeros(1, device=dev)
    root_state = robot.data.default_root_state.clone()
    root_state[0, :3] = torch.tensor([0.0, 0.0, 0.2955], device=dev)
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
    sim.set_camera_view(eye=[-2.5, -3.0, 2.5], target=[1.5, 0.0, 0.0])

    robot = design_scene()

    sim.reset()

    # ── OmniLidar 센서 생성 (Isaac Sim 5.0+ 공식 방식) ────────────────────────
    # LidarRtx 생성자는 내부적으로 IsaacSensorCreateRtxLidar 커맨드를 호출해
    # OmniLidar prim 과 render product 를 자동 생성한다.
    #
    # translation 을 명시적으로 지정해야 set_local_pose() 경로로 분기된다.
    # 미지정 시 LidarRtx 내부에서 position=[0,0,0] 이 설정되어
    # set_world_pose([0,0,0]) 가 호출되므로 센서가 지면(z=0)으로 이동한다.
    #
    # prim_path 를 base_link 아래로 지정하면 로봇이 이동해도 USD 계층을 통해
    # 센서가 자동으로 추종한다.
    lidar = LidarRtx(
        prim_path=LIDAR_PRIM,
        name=f"lidar_{args_cli.lidar.replace('-', '_')}",
        translation=np.array([0.0, 0.0, LIDAR_Z]),
        config_file_name=LIDAR_CONFIG,
    )
    _rtx_note = _lidar_spec.get("rtx_note", "")
    print(f"[LiDAR] OmniLidar 생성: {LIDAR_PRIM}  모델={args_cli.lidar}  config={LIDAR_CONFIG}")
    print(f"[LiDAR] 스펙: {_lidar_channels}ch / 10Hz / 2048res / PPS={_lidar_pps:,} / 최대 120m")
    if _rtx_note:
        print(f"[LiDAR] ※ {_rtx_note}")

    simulation_app.update()
    simulation_app.update()

    # ── 센서 초기화 + 어노테이터 ──────────────────────────────────────────────
    lidar.initialize()

    # 매 렌더 프레임의 부분 포인트 클라우드 (N×3 월드 좌표) — 시각화·통계 공용
    lidar.attach_annotator(LIDAR_ANNOT)

    # ── debug_draw 인터페이스 (직접 누적 시각화) ──────────────────────────────
    # RtxLidarDebugDrawPointCloudBuffer writer 는 새 스캔 누적 시작 시
    # 화면을 지워 번쩍거림을 유발한다.
    # 대신 Python 에서 직접 누적 후, 1회전 분량이 쌓이면 atomic 교체하여
    # 이전 포인트 클라우드가 항상 화면에 유지되도록 한다.
    _dd       = _dd_module.acquire_debug_draw_interface()
    _pts_buf  = []          # 누적 중인 부분 스캔 청크 리스트
    _pts_show = None        # 현재 화면에 표시 중인 완성 스캔 (np.ndarray)
    print(f"[LiDAR] annotator: {LIDAR_ANNOT}  |  debug_draw 직접 시각화 활성화")

    # ── 관절 인덱스 ────────────────────────────────────────────────────────────
    jnames      = robot.joint_names
    idx_steer_l = jnames.index(STEER_LEFT_NAME)
    idx_steer_r = jnames.index(STEER_RIGHT_NAME)
    idx_wheel_l = jnames.index(WHEEL_LEFT_NAME)
    idx_wheel_r = jnames.index(WHEEL_RIGHT_NAME)

    print(f"\n[INFO] 관절 ({len(jnames)}개): {jnames}")

    acker      = HunterSEAckermann(device=robot.device)
    controller = HunterKeyboardController(max_speed=SPEED_LEVELS[DEFAULT_SPEED_KEY])

    prev_delta   = 0.0
    v_cmd        = 0.0
    delta_smooth = 0.0
    step         = 0

    print("\n[텔레오프] 시작. 뷰포트에 포커스 후 키보드 입력.\n")

    while simulation_app.is_running():

        if controller.quit_requested:
            break

        if controller.reset_requested:
            reset_robot(robot, sim)
            prev_delta = delta_smooth = v_cmd = 0.0
            step = 0
            continue

        # 키 명령 + 조향 rate limiting
        v_target, delta_target = controller.get_command()
        max_change   = MAX_STEER_RATE * STEP_DT
        delta_smooth = prev_delta + max(
            -max_change, min(max_change, delta_target - prev_delta)
        )
        prev_delta = delta_smooth
        v_cmd      = v_target

        # Ackermann 계산 → 관절 명령
        steer_l, steer_r, omega_l, omega_r = acker.compute_scalar(v_cmd, delta_smooth)
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

        sim.step()
        robot.update(STEP_DT)
        step += 1

        # ── HUD ───────────────────────────────────────────────────────────────
        if step % HUD_INTERVAL == 0:
            pos = robot.data.root_pos_w[0]
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            spd = float(robot.data.root_lin_vel_b[0].norm())
            print(
                f"\r[{step*STEP_DT:6.1f}s] "
                f"pos=({pos[0].item():6.2f},{pos[1].item():6.2f}) "
                f"yaw={math.degrees(float(yaw[0])):6.1f}°  "
                f"spd={spd:.2f}m/s  "
                f"cmd v={v_cmd:+.2f} δ={math.degrees(delta_smooth):+5.1f}°   ",
                end="",
                flush=True,
            )

        # ── LiDAR 포인트 클라우드 누적 + 시각화 ──────────────────────────────
        # OGN 문서 확인: IsaacExtractRTXSensorPointCloud 는 센서 로컬 좌표 반환.
        # info["transform"] = 4×4 lidar→world 변환 행렬 (row-major, USD 규약).
        # draw_points 는 월드 좌표를 요구하므로 반드시 변환 적용이 필요하다.
        frame = lidar.get_current_frame()
        raw = frame.get(LIDAR_ANNOT, None)
        if raw is not None:
            pts_local = raw.get("data", None) if isinstance(raw, dict) else raw
            transform  = raw.get("info", {}).get("transform", None) if isinstance(raw, dict) else None

            if pts_local is not None and hasattr(pts_local, "__len__") and len(pts_local) > 0:
                pts_local = np.asarray(pts_local, dtype=np.float64)
                if pts_local.ndim == 2 and pts_local.shape[1] >= 3:
                    # 유효 포인트 (센서 기준 거리 > 5 cm, NaN 제외)
                    dist_l = np.linalg.norm(pts_local[:, :3], axis=1)
                    valid  = np.isfinite(dist_l) & (dist_l > 0.05)
                    chunk_local = pts_local[valid, :3]

                    if len(chunk_local) > 0 and transform is not None:
                        # 로컬 → 월드 변환 (row-major: world = local_homo @ T)
                        T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
                        ones = np.ones((len(chunk_local), 1), dtype=np.float64)
                        homo = np.concatenate([chunk_local, ones], axis=1)  # Nx4
                        chunk_world = (homo @ T)[:, :3].astype(np.float32)

                        _pts_buf.append(chunk_world)
                        total = sum(len(c) for c in _pts_buf)

                        if total >= LIDAR_PTS_PER_SCAN:
                            # 1회전 완성 → atomic 교체 (clear → draw 사이 빈 프레임 없음)
                            _pts_show = np.vstack(_pts_buf)
                            _pts_buf.clear()

                            _dd.clear_points()
                            n = len(_pts_show)
                            pts_list = _pts_show[:, :3].tolist()
                            _dd.draw_points(
                                pts_list,
                                [(0.0, 1.0, 0.0, 1.0)] * n,  # RGBA 녹색
                                [2.0] * n,
                            )

        # ── LiDAR 콘솔 통계 (센서 로컬 좌표로 range 계산) ─────────────────────
        if step % LIDAR_PRINT_INTERVAL == 0:
            stats = _lidar_stats(frame)
            print(f"\n  [LiDAR] {stats}")

    _dd.clear_points()
    lidar.detach_all_annotators()
    print("\n[종료] 시뮬레이션 종료.")


if __name__ == "__main__":
    main()
    simulation_app.close()
