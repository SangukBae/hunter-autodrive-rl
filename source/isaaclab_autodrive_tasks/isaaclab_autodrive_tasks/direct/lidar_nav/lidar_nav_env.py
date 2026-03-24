# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR 기반 자율주행 환경 — goal-reaching + obstacle avoidance.

Isaac Lab DirectRLEnv 기반. RTX OmniLidar (OS1-32) + 16×16m 벽+장애물 맵.

관측 (82D):
    lidar[0:80]  : 80-sector 2D LiDAR (정규화: dist / lidar_range, [0, 1])
    goal_dist    : 로봇 → 목표 거리 (정규화: dist / map_size)
    goal_angle   : 로봇 로컬 프레임 기준 목표 방향각 / π → [-1, 1]

행동 (2D):  [-1, 1] 정규화
    linear_vel  : 선속도  → [0, max_linear_vel] m/s (전진만)
    angular_vel : 각속도  → [-max_angular_vel, +max_angular_vel] rad/s

LiDAR 방식:
    RTX OmniLidar (OS1-32, teleop 동일 방식) — 물리 기반 광자 추적.
    OS1-32 @ 10Hz → 1회전 = 20 physics step ≈ 5 RL step.
    각 step에서 수신된 부분 스캔(~72°)을 롤링-min 버퍼에 누적하여 80-sector 맵 갱신.
    버퍼는 에피소드 리셋 시에만 초기화됨.

충돌 판정:
    LiDAR 최소 감지 거리 < lidar_collision_threshold → 충돌 종료.
    벽·장애물 통합 판정 (물리 메시 기반, 별도 geometry 계산 없음).

맵:
    16×16m 물리 벽 경계 + kinematic cylinder 장애물 × num_obstacles.
    매 에피소드 로봇/목표/장애물 랜덤 배치.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from .lidar_nav_env_cfg import LidarNavEnvCfg

import sys
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se_v0.ackermann import HunterSEAckermann, MAX_STEER as _MAX_STEER
from hunter_se_v0.lidar_cfg import get_lidar_spec

# 로봇 근사 반경 (충돌 판정용)
# Hunter SE: rear_track=0.504m → half-width≈0.35m, wheelbase=0.548m → half-length≈0.40m
# 차체 + 바퀴 여유 포함 보수적 bounding circle 반경
_ROBOT_RADIUS = 0.45    # [m]

# RTX 모드에서 사용하는 어노테이터 (teleop 와 동일)
_RTX_ANNOT = "IsaacExtractRTXSensorPointCloudNoAccumulator"
# RTX LiDAR base_link 기준 마운트 높이
_RTX_LIDAR_Z = 0.2      # [m]


class LidarNavEnv(DirectRLEnv):
    """LiDAR 자율주행 환경.

    Phase C / Phase E 공통 클래스. cfg.use_walls, cfg.use_physical_obstacles
    플래그로 Phase 전환.
    """

    cfg: LidarNavEnvCfg

    def __init__(self, cfg: LidarNavEnvCfg, render_mode: str | None = None, **kwargs):
        # 물리 장애물 오브젝트 리스트 (Phase E; _setup_scene에서 채워짐)
        self._obstacle_objects: list[RigidObject] = []
        # RTX 모드 전용 (sim.reset() 이후 _init_rtx_lidars 에서 채워짐)
        self._rtx_lidars: list = []
        self._rtx_sector_buf: torch.Tensor | None = None

        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스 캐싱
        self._steer_l_idx, _ = self.robot.find_joints("fr_steer_left_joint")
        self._steer_r_idx, _ = self.robot.find_joints("fr_steer_right_joint")
        self._wheel_l_idx, _ = self.robot.find_joints("re_left_joint")
        self._wheel_r_idx, _ = self.robot.find_joints("re_right_joint")

        # Ackermann 처리기 (HunterSEAckermann, 중심각 기준)
        self._ackermann = HunterSEAckermann(device=self.device)

        # 이전 스텝 조향각 (rate limiting용)
        self._prev_delta = torch.zeros(self.num_envs, device=self.device)

        # 목표 위치 버퍼 (월드 좌표 XY)
        self._goal_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # 이전 스텝 goal 거리 (progress 보상 계산용)
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)

        # 장애물 위치·반경 버퍼 (물리 장애물 배치에 사용)
        self._obstacle_pos   = torch.zeros(self.num_envs, self.cfg.num_obstacles, 2, device=self.device)
        self._obstacle_radii = torch.zeros(self.num_envs, self.cfg.num_obstacles,    device=self.device)

        # LiDAR 기반 충돌 판정용 — _get_observations() 에서 매 스텝 갱신
        self._last_lidar_obs: torch.Tensor | None = None

        # RTX LiDAR 초기화 (sim.reset() 완료 후 여기서 생성해야 함)
        if cfg.lidar_use_rtx:
            self._init_rtx_lidars()
            self._rtx_sector_buf = torch.full(
                (self.num_envs, cfg.num_sectors),
                cfg.lidar_range,
                device=self.device,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 씬 설정
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        # ── 로봇 ──────────────────────────────────────────────────────────────
        self.robot = Articulation(self.cfg.robot)

        # ── 평지 (전역, 복제 제외) ─────────────────────────────────────────────
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ── 물리 장애물 실린더 ──────────────────────────────────────────────────
        self._obstacle_objects = []
        for i in range(self.cfg.num_obstacles):
            obs_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self.cfg.obstacle_radius_max,
                    height=self.cfg.obstacle_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.3, 0.1), metallic=0.1
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(i * 3 - 6), 0.0, self.cfg.obstacle_height / 2)
                ),
            )
            self._obstacle_objects.append(RigidObject(obs_cfg))

        # ── Phase E: 물리 벽 스폰 (clone 이전 → 자동으로 전체 env에 복제됨) ──
        if self.cfg.use_walls:
            self._spawn_walls()

        # ── RTX LiDAR: clone 이전에 env_0 에 OmniLidar prim 생성 ──────────────
        # copy_from_source=False 라이브 미러 방식에서 clone 이후 env_0 에 prim 을 추가하면
        # env_1+ 에는 Xform(래퍼)만 복사되어 OmniLidar 타입 체크에서 실패한다.
        # clone 이전 생성 → clone 이 올바른 OmniLidar 복사본을 만든다.
        if self.cfg.lidar_use_rtx:
            self._spawn_rtx_lidar_prim_env0()

        # ── 환경 복제 ─────────────────────────────────────────────────────────
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # ── 씬에 등록 ─────────────────────────────────────────────────────────
        self.scene.articulations["robot"] = self.robot
        for i, obs_obj in enumerate(self._obstacle_objects):
            self.scene.rigid_objects[f"obstacle_{i}"] = obs_obj

        # ── LiDAR 센서 ──────────────────────────────────────────────────────
        if self.cfg.lidar_use_rtx:
            # RTX 모드: prim 은 clone 이전(_spawn_rtx_lidar_prim_env0)에 이미 생성됨.
            # LidarRtx 래퍼 초기화는 sim.reset() 완료 후 _init_rtx_lidars() 에서 수행.
            pass
        else:
            # Warp 해석적 LiDAR (MultiMeshRayCaster) — channels × 80섹터 ray.
            # clone 이후 생성해야 모든 메시 prim 이 존재함.
            _lidar_channels = get_lidar_spec(self.cfg.lidar_model)["channels"]
            mesh_targets: list = ["/World/ground"]
            for i in range(self.cfg.num_obstacles):
                mesh_targets.append(
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr=f"/World/envs/env_.*/Obstacle_{i}",
                        track_mesh_transforms=True,
                        is_shared=False,
                    )
                )
            if self.cfg.use_walls:
                for wall_name in ["Wall_N", "Wall_S", "Wall_E", "Wall_W"]:
                    mesh_targets.append(
                        MultiMeshRayCasterCfg.RaycastTargetCfg(
                            prim_expr=f"/World/envs/env_.*/{wall_name}",
                            track_mesh_transforms=False,
                            is_shared=True,
                        )
                    )
            lidar_cfg = MultiMeshRayCasterCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
                mesh_prim_paths=mesh_targets,
                pattern_cfg=patterns.LidarPatternCfg(
                    channels=_lidar_channels,
                    vertical_fov_range=(-22.5, 22.5),
                    horizontal_fov_range=(-180.0, 180.0),
                    horizontal_res=4.5,   # 360° / 80섹터 = 4.5°/섹터
                ),
                max_distance=self.cfg.lidar_range,
                ray_alignment="base",
                debug_vis=self.cfg.lidar_debug_vis,
            )
            self._lidar_sensor = MultiMeshRayCaster(lidar_cfg)
            self.scene.sensors["lidar"] = self._lidar_sensor

        # ── 조명 ──────────────────────────────────────────────────────────────
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_walls(self):
        """env_0 로컬 좌표에 4개 정적 충돌 벽을 스폰.

        clone_environments() 이전에 호출해야 함.
        복제 시 자동으로 모든 env에 반영됨.
        벽은 단순 static collider (RigidBody 없음).
        """
        m = self.cfg.map_size
        t = self.cfg.wall_thickness
        h = self.cfg.wall_height
        L = 2.0 * m + 2.0 * t   # 벽 길이 (모서리 포함)

        wall_cfg_ns = sim_utils.CuboidCfg(   # N/S 벽 (x 방향으로 긴 벽)
            size=(L, t, h),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.65), metallic=0.0
            ),
        )
        wall_cfg_ew = sim_utils.CuboidCfg(   # E/W 벽 (y 방향으로 긴 벽)
            size=(t, L, h),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.65), metallic=0.0
            ),
        )

        # copy_from_source=False 이므로 env_1+ 는 env_0 을 미러링한다.
        # env_0 에만 스폰하면 나머지 env 에도 자동으로 반영된다.
        # 로컬 좌표계(env_0 원점 기준) 사용 → 각 env 의 world transform 이 올바른 위치를 만든다.
        z = h / 2.0
        base = "/World/envs/env_0"

        # North (y = +m)
        wall_cfg_ns.func(f"{base}/Wall_N", wall_cfg_ns, translation=(0.0, m + t / 2, z))
        # South (y = -m)
        wall_cfg_ns.func(f"{base}/Wall_S", wall_cfg_ns, translation=(0.0, -(m + t / 2), z))
        # East (x = +m)
        wall_cfg_ew.func(f"{base}/Wall_E", wall_cfg_ew, translation=(m + t / 2, 0.0, z))
        # West (x = -m)
        wall_cfg_ew.func(f"{base}/Wall_W", wall_cfg_ew, translation=(-(m + t / 2), 0.0, z))

    def _spawn_rtx_lidar_prim_env0(self) -> None:
        """clone_environments() 이전에 env_0 에 OmniLidar prim 을 생성한다.

        copy_from_source=False 라이브 미러 방식에서는 clone 이후 env_0 에 prim 을 추가하면
        env_1+ 에 Xform(래퍼)만 전파되어 OmniLidar 타입 체크 실패가 발생한다.
        clone 이전에 생성해야 clone 이 올바른 OmniLidar 복사본을 만든다.

        _spawn_walls() 패턴과 동일.
        """
        import omni.kit.commands
        from pxr import Gf

        spec = get_lidar_spec(self.cfg.lidar_model)
        rtx_config = spec["rtx_config"]

        print(f"[LidarNavEnv] RTX LiDAR prim 생성 (env_0, clone 이전): config={rtx_config}")

        omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/World/envs/env_0/Robot/base_link/Lidar",
            config=rtx_config,
            translation=Gf.Vec3d(0.0, 0.0, _RTX_LIDAR_Z),
            parent=None,
        )

    def _init_rtx_lidars(self) -> None:
        """RTX OmniLidar 센서를 각 환경 로봇에 부착한다.

        sim.reset() 완료 후 호출해야 USD 스테이지가 준비된 상태.
        OmniLidar prim 은 _spawn_rtx_lidar_prim_env0() + clone 으로 이미 존재하므로
        USD PrimRange 로 탐색 후 LidarRtx 래퍼를 씌운다.

        성능 주의: 환경당 GPU render product 1개 생성 → num_envs 1~8 권장.
        """
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.sensors.rtx import LidarRtx
        from pxr import Usd

        enable_extension("isaacsim.util.debug_draw")

        spec = get_lidar_spec(self.cfg.lidar_model)

        print(
            f"[LidarNavEnv] RTX LiDAR 초기화: {self.num_envs}개 환경 × {spec['channels']}ch"
        )
        if self.num_envs > 8:
            print(
                f"[LidarNavEnv] 경고: RTX 모드에서 num_envs={self.num_envs} 는 매우 느릴 수 있습니다."
                "  1~8 을 권장합니다."
            )

        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):
            xform_path = f"/World/envs/env_{i}/Robot/base_link/Lidar"
            xform_prim = stage.GetPrimAtPath(xform_path)
            if not xform_prim.IsValid():
                raise RuntimeError(
                    f"[LidarNavEnv] RTX LiDAR Xform prim 이 없습니다: {xform_path}\n"
                    "  _spawn_rtx_lidar_prim_env0() 가 clone_environments() 이전에 "
                    "호출되었는지 확인하세요."
                )

            # IsaacSensorCreateRtxLidar 는 요청 경로에 Xform 을 만들고
            # 그 안에 OmniLidar 자식 prim 을 생성한다. 자식을 탐색한다.
            omni_lidar_path: str | None = None
            for child in Usd.PrimRange(xform_prim):
                if child.GetTypeName() == "OmniLidar":
                    omni_lidar_path = str(child.GetPath())
                    break

            if omni_lidar_path is None:
                raise RuntimeError(
                    f"[LidarNavEnv] {xform_path} 하위에 OmniLidar prim 이 없습니다.\n"
                    f"  prim 타입: {xform_prim.GetTypeName()}"
                )

            lidar = LidarRtx(
                prim_path=omni_lidar_path,
                name=f"rtx_lidar_{i}",
            )
            lidar.initialize()
            lidar.attach_annotator(_RTX_ANNOT)
            self._rtx_lidars.append(lidar)

        # RTX 렌더 파이프라인 준비 대기
        for _ in range(4):
            self.sim.app.update()

        print(f"[LidarNavEnv] RTX LiDAR 초기화 완료.")

    # ─────────────────────────────────────────────────────────────────────────
    # 행동 전처리
    # ─────────────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """정규화된 행동 [-1, 1] → 실제 제어값으로 변환.

        HunterSEAckermann (WheeledLab MuSHR 방식) 사용:
            - 조향각 기준: 차체 중심(bicycle model) 조향각 delta_c
            - 외륜/내륜 각도 및 후륜 차동 속도를 클래스가 일괄 계산
        """
        self._actions = actions.clone()

        # linear_vel: [-1, 1] → [0, max_linear_vel]  (전진 전용)
        lin_vel = (actions[:, 0] + 1.0) / 2.0 * self.cfg.max_linear_vel

        # action[1]: [-1, 1] → 차체 중심 조향각 delta_c [-MAX_STEER, MAX_STEER]
        delta_target = actions[:, 1] * _MAX_STEER

        # 조향 변화율 제한: 실제 Hunter SE 조향 속도 ≈ 0.5 rad/s
        _MAX_STEER_RATE = 0.5 * self.step_dt
        delta = self._prev_delta + torch.clamp(
            delta_target - self._prev_delta, -_MAX_STEER_RATE, _MAX_STEER_RATE
        )
        self._prev_delta = delta.detach()

        # HunterSEAckermann: 중심각 → 좌/우 조향각 + 후륜 차동 각속도
        self._steer_l, self._steer_r, self._omega_l, self._omega_r = (
            self._ackermann.compute(lin_vel, delta)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 행동 적용
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self._steer_l.unsqueeze(-1), joint_ids=self._steer_l_idx
        )
        self.robot.set_joint_position_target(
            self._steer_r.unsqueeze(-1), joint_ids=self._steer_r_idx
        )
        self.robot.set_joint_velocity_target(
            self._omega_l.unsqueeze(-1), joint_ids=self._wheel_l_idx
        )
        self.robot.set_joint_velocity_target(
            self._omega_r.unsqueeze(-1), joint_ids=self._wheel_r_idx
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 관측 계산
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """80-sector LiDAR + goal_dist + goal_angle."""
        lidar_obs = self._compute_lidar()
        self._last_lidar_obs = lidar_obs  # LiDAR 기반 충돌 판정용 캐시

        robot_pos = self.robot.data.root_pos_w[:, :2]
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        dx = self._goal_pos[:, 0] - robot_pos[:, 0]
        dy = self._goal_pos[:, 1] - robot_pos[:, 1]

        # goal 거리 정규화 [0, 1]
        goal_dist = torch.sqrt(dx * dx + dy * dy) / self.cfg.map_size

        # goal 각도 (로봇 로컬, 정규화 [-1, 1])
        goal_angle_w = torch.atan2(dy, dx)
        goal_angle_r = (goal_angle_w - yaw) / math.pi
        goal_angle_r = ((goal_angle_r + 1.0) % 2.0) - 1.0

        obs = torch.cat(
            [lidar_obs, goal_dist.unsqueeze(1), goal_angle_r.unsqueeze(1)], dim=1
        )
        return {"policy": obs}

    # ─────────────────────────────────────────────────────────────────────────
    # 보상 계산
    # ─────────────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """goal_progress + goal_reached + collision + proximity + time_penalty."""
        robot_pos = self.robot.data.root_pos_w[:, :2]
        dx = self._goal_pos[:, 0] - robot_pos[:, 0]
        dy = self._goal_pos[:, 1] - robot_pos[:, 1]
        curr_dist = torch.sqrt(dx * dx + dy * dy)

        # dense progress
        progress = (self._prev_goal_dist - curr_dist) * self.cfg.progress_scale
        self._prev_goal_dist = curr_dist.clone()

        # sparse goal
        goal_reached = (curr_dist < self.cfg.goal_threshold).float() * self.cfg.goal_reward

        # sparse collision
        collision = self._get_collision_mask().float() * self.cfg.collision_penalty

        # smooth proximity
        min_dist = self._get_min_obstacle_dist()
        prox_ratio = torch.clamp(1.0 - min_dist / self.cfg.proximity_threshold, min=0.0)
        proximity = -self.cfg.proximity_penalty_max * prox_ratio

        # time penalty
        time_pen = torch.full((self.num_envs,), self.cfg.time_penalty, device=self.device)

        return progress + goal_reached + collision + proximity + time_pen

    # ─────────────────────────────────────────────────────────────────────────
    # 종료 조건
    # ─────────────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self.robot.data.root_pos_w[:, :2]
        dx = self._goal_pos[:, 0] - robot_pos[:, 0]
        dy = self._goal_pos[:, 1] - robot_pos[:, 1]
        curr_dist = torch.sqrt(dx * dx + dy * dy)

        goal_reached = curr_dist < self.cfg.goal_threshold
        collision    = self._get_collision_mask()

        # 물리 벽이 map 경계를 막으므로 OOB 체크 불필요
        terminated = goal_reached | collision

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ─────────────────────────────────────────────────────────────────────────
    # 에피소드 리셋
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n       = len(env_ids)
        origins = self.scene.env_origins[env_ids]   # (n, 3)

        # ── 로봇: 맵 내 랜덤 위치 + 랜덤 yaw ─────────────────────────────────
        spawn_range = self.cfg.map_size - 1.0
        rx   = torch.empty(n, device=self.device).uniform_(-spawn_range, spawn_range)
        ry   = torch.empty(n, device=self.device).uniform_(-spawn_range, spawn_range)
        ryaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        zeros = torch.zeros(n, device=self.device)

        robot_spawn_z = self.cfg.robot.init_state.pos[2]   # 0.2955 m (바퀴 접지 높이)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0] = origins[:, 0] + rx
        root_state[:, 1] = origins[:, 1] + ry
        root_state[:, 2] = origins[:, 2] + robot_spawn_z  # 지형 원점 + 바퀴 접지 높이
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, ryaw)
        root_state[:, 7:]  = 0.0

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            None, env_ids,
        )

        # ── 목표 위치: 맵 내 랜덤, 로봇에서 2m 이상 ──────────────────────────
        goal_range = self.cfg.map_size - 0.5
        gx = torch.empty(n, device=self.device).uniform_(-goal_range, goal_range)
        gy = torch.empty(n, device=self.device).uniform_(-goal_range, goal_range)
        robot_local = torch.stack([rx, ry], dim=1)
        goal_local  = torch.stack([gx, gy], dim=1)
        too_close = torch.norm(goal_local - robot_local, dim=1) < 2.0
        goal_local[too_close, 0] = torch.clamp(
            goal_local[too_close, 0] + 2.0, -goal_range, goal_range
        )
        self._goal_pos[env_ids, 0] = origins[:, 0] + goal_local[:, 0]
        self._goal_pos[env_ids, 1] = origins[:, 1] + goal_local[:, 1]

        # ── 장애물 위치·반경 랜덤화 (로봇으로부터 최소 거리 보장) ──────────────
        obs_range  = self.cfg.map_size - 0.5
        # 안전 이격: 로봇 반경 + 장애물 최대 반경 + 여유 0.5 m
        min_sep    = _ROBOT_RADIUS + self.cfg.obstacle_radius_max + 0.5
        robot_local = torch.stack([rx, ry], dim=1)   # (n, 2)  env 로컬 좌표

        for i in range(self.cfg.num_obstacles):
            # 최대 20회 재시도 — 모든 env가 조건을 만족하면 바로 종료
            for _ in range(20):
                ox = torch.empty(n, device=self.device).uniform_(-obs_range, obs_range)
                oy = torch.empty(n, device=self.device).uniform_(-obs_range, obs_range)
                dist_to_robot = torch.norm(
                    torch.stack([ox, oy], dim=1) - robot_local, dim=1
                )
                if (dist_to_robot >= min_sep).all():
                    break

            self._obstacle_pos[env_ids, i, 0] = origins[:, 0] + ox
            self._obstacle_pos[env_ids, i, 1] = origins[:, 1] + oy
            self._obstacle_radii[env_ids, i] = torch.empty(n, device=self.device).uniform_(
                self.cfg.obstacle_radius_min, self.cfg.obstacle_radius_max
            )

        # ── 물리 장애물 위치 리셋 ──────────────────────────────────────────────
        for i, obs_obj in enumerate(self._obstacle_objects):
            new_state = obs_obj.data.default_root_state[env_ids].clone()
            new_state[:, 0] = self._obstacle_pos[env_ids, i, 0]
            new_state[:, 1] = self._obstacle_pos[env_ids, i, 1]
            new_state[:, 2] = origins[:, 2] + self.cfg.obstacle_height / 2.0
            new_state[:, 7:] = 0.0
            obs_obj.write_root_pose_to_sim(new_state[:, :7], env_ids)
            obs_obj.write_root_velocity_to_sim(new_state[:, 7:], env_ids)

        # ── RTX 롤링 버퍼 초기화 (RTX 모드 전용) ──────────────────────────────
        if self.cfg.lidar_use_rtx and self._rtx_sector_buf is not None:
            self._rtx_sector_buf[env_ids] = self.cfg.lidar_range

        # ── 조향 rate limiter 초기화 ──────────────────────────────────────────
        self._prev_delta[env_ids] = 0.0

        # ── prev_goal_dist 초기화 ─────────────────────────────────────────────
        ddx = self._goal_pos[env_ids, 0] - self.robot.data.root_pos_w[env_ids, 0]
        ddy = self._goal_pos[env_ids, 1] - self.robot.data.root_pos_w[env_ids, 1]
        self._prev_goal_dist[env_ids] = torch.sqrt(ddx * ddx + ddy * ddy)

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼: LiDAR (Warp / RTX 공통 디스패처)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_lidar(self) -> torch.Tensor:
        """cfg.lidar_use_rtx 에 따라 RTX 또는 Warp LiDAR 결과를 반환."""
        if self.cfg.lidar_use_rtx:
            return self._compute_lidar_rtx()
        return self._compute_lidar_warp()

    def _compute_lidar_warp(self) -> torch.Tensor:
        """Warp MultiMeshRayCaster → 80섹터 2D 거리맵 [0, 1].

        C채널 × S수평섹터 ray. 각 수평섹터에서 C채널 최솟값 → (N, S).
        """
        D = self.cfg.lidar_range
        C = get_lidar_spec(self.cfg.lidar_model)["channels"]
        S = self.cfg.num_sectors

        pos_w  = self._lidar_sensor.data.pos_w
        hits_w = self._lidar_sensor.data.ray_hits_w

        diff      = hits_w - pos_w.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)
        distances = distances.nan_to_num(posinf=D).clamp(max=D)
        lidar_2d  = distances.view(self.num_envs, C, S).min(dim=1).values
        return lidar_2d / D

    def _compute_lidar_rtx(self) -> torch.Tensor:
        """RTX OmniLidar 부분 스캔 → 롤링-min 80섹터 2D 거리맵 [0, 1].

        OS1-32 @ 10Hz 기준: 1회전 = 20 physics step ≈ 5 RL step.
        매 RL step 에서 수신된 부분 스캔(~72°)을 _rtx_sector_buf 에 min 으로 누적.
        버퍼는 에피소드 리셋(_reset_idx) 시에만 초기화됨.

        좌표: IsaacExtractRTXSensorPointCloudNoAccumulator 는 센서 로컬 프레임 반환.
              azimuth = atan2(y, x) [-180°, +180°] → sector_idx.
        """
        D = self.cfg.lidar_range
        S = self.cfg.num_sectors

        for i in range(self.num_envs):
            frame = self._rtx_lidars[i].get_current_frame()
            raw = frame.get(_RTX_ANNOT, None)
            if raw is None:
                continue

            pts_local = raw.get("data", None) if isinstance(raw, dict) else raw
            if pts_local is None or (hasattr(pts_local, "__len__") and len(pts_local) == 0):
                continue

            pts = np.asarray(pts_local, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                continue

            dist = np.linalg.norm(pts[:, :3], axis=1)
            valid = np.isfinite(dist) & (dist > 0.05)
            if not valid.any():
                continue

            pts_v  = pts[valid, :3]
            dist_v = dist[valid].clip(max=D)

            # 센서 로컬 방위각 → 섹터 인덱스
            azi_deg    = np.degrees(np.arctan2(pts_v[:, 1], pts_v[:, 0]))  # [-180, 180]
            sector_idx = ((azi_deg + 180.0) * (S / 360.0)).astype(np.int32) % S

            # 섹터별 최솟값 (numpy in-place scatter-min)
            sector_dists = np.full(S, D, dtype=np.float32)
            np.minimum.at(sector_dists, sector_idx, dist_v)

            # 롤링 버퍼 min 업데이트
            self._rtx_sector_buf[i] = torch.minimum(
                self._rtx_sector_buf[i],
                torch.from_numpy(sector_dists).to(self.device),
            )

        return self._rtx_sector_buf / D

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼: 충돌 판정
    # ─────────────────────────────────────────────────────────────────────────

    def _get_collision_mask(self) -> torch.Tensor:
        """LiDAR 최소 감지 거리 < lidar_collision_threshold 이면 충돌 (N,) bool.

        RTX LiDAR 포인트 클라우드 기반 — 물리 메시와 직접 상호작용하므로
        벽·장애물 모두 통합 판정된다.
        """
        if self._last_lidar_obs is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        min_dist_m = self._last_lidar_obs.min(dim=1).values * self.cfg.lidar_range
        return min_dist_m < self.cfg.lidar_collision_threshold

    def _get_min_obstacle_dist(self) -> torch.Tensor:
        """LiDAR 최소 감지 거리 [m] — 근접 페널티 계산용."""
        if self._last_lidar_obs is None:
            return torch.full((self.num_envs,), self.cfg.lidar_range, device=self.device)
        return self._last_lidar_obs.min(dim=1).values * self.cfg.lidar_range
