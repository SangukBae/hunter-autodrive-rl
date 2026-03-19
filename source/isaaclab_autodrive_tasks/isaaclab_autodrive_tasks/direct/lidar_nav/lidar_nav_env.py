# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR 기반 자율주행 환경 — goal-reaching + obstacle avoidance.

Isaac Lab DirectRLEnv 기반.

관측 (82D):
    lidar[0:80]  : 80-sector 2D LiDAR (정규화: dist / lidar_range, [0, 1])
    goal_dist    : 로봇 → 목표 거리 (정규화: dist / map_size)
    goal_angle   : 로봇 로컬 프레임 기준 목표 방향각 / π → [-1, 1]

행동 (2D):  [-1, 1] 정규화
    linear_vel  : 선속도  → [0, max_linear_vel] m/s (전진만)
    angular_vel : 각속도  → [-max_angular_vel, +max_angular_vel] rad/s

Phase C (기본):
    - 해석적 2D LiDAR (원통 교차)
    - 가상 장애물 위치 버퍼 (물리 프림 없음)
    - goal/장애물/로봇 에피소드마다 랜덤화

Phase E (use_walls=True, use_physical_obstacles=True):
    - 물리 벽 (4개 static cuboid)
    - 물리 장애물 (kinematic cylinder RigidObject × num_obstacles)
    - 해석적 LiDAR: 원통 + 벽 동시 감지
    - 벽 물리 충돌로 map 이탈 방지
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from .lidar_nav_env_cfg import LidarNavEnvCfg

import sys
sys.path.insert(0, "/workspace/hunter_autodrive")
from hunter_se.ackermann import HunterSEAckermann, MAX_STEER as _MAX_STEER

# 로봇 근사 반경 (충돌 판정용)
_ROBOT_RADIUS = 0.30    # [m]


class LidarNavEnv(DirectRLEnv):
    """LiDAR 자율주행 환경.

    Phase C / Phase E 공통 클래스. cfg.use_walls, cfg.use_physical_obstacles
    플래그로 Phase 전환.
    """

    cfg: LidarNavEnvCfg

    def __init__(self, cfg: LidarNavEnvCfg, render_mode: str | None = None, **kwargs):
        # 물리 장애물 오브젝트 리스트 (Phase E; _setup_scene에서 채워짐)
        self._obstacle_objects: list[RigidObject] = []

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

        # 장애물 위치·반경 버퍼 (해석적 LiDAR 및 충돌 판정에 사용)
        self._obstacle_pos   = torch.zeros(self.num_envs, self.cfg.num_obstacles, 2, device=self.device)
        self._obstacle_radii = torch.zeros(self.num_envs, self.cfg.num_obstacles,    device=self.device)

        # 섹터 각도 오프셋: sector i → local angle = i * 2π/S (0=전방)
        S = self.cfg.num_sectors
        self._sector_offsets = torch.arange(S, device=self.device) * (2.0 * math.pi / S)

    # ─────────────────────────────────────────────────────────────────────────
    # 씬 설정
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        # ── 로봇 ──────────────────────────────────────────────────────────────
        self.robot = Articulation(self.cfg.robot)

        # ── 평지 (전역, 복제 제외) ─────────────────────────────────────────────
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ── Phase E: 물리 장애물 실린더 ──────────────────────────────────────
        if self.cfg.use_physical_obstacles:
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

        # ── 환경 복제 ─────────────────────────────────────────────────────────
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # ── Phase E: 물리 벽 스폰 (복제 이후 → env origin 알아야 불필요, env_0 기준으로 미리 스폰) ──
        # 실제로는 복제 전에 env_0 아래 스폰하면 자동으로 복제됨.
        # 하지만 env_0 기준 좌표계 = 로컬 좌표이므로, 복제 후에 스폰 시 env_origin 더함.
        # 여기서는 env_origins가 확정된 시점(복제 직후)에 per-env 스폰.
        if self.cfg.use_walls:
            self._spawn_walls_per_env()

        # ── 씬에 등록 ─────────────────────────────────────────────────────────
        self.scene.articulations["robot"] = self.robot
        for i, obs_obj in enumerate(self._obstacle_objects):
            self.scene.rigid_objects[f"obstacle_{i}"] = obs_obj

        # ── 조명 ──────────────────────────────────────────────────────────────
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_walls_per_env(self):
        """각 env에 4개의 정적 충돌 벽을 스폰.

        환경 복제 이후에 env_origins를 이용해 월드 좌표로 직접 스폰.
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
        """80-sector 해석적 LiDAR + goal_dist + goal_angle."""
        lidar_obs = self._compute_lidar()

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

        if self.cfg.use_walls:
            # 물리 벽이 있으면 로봇이 map을 벗어날 수 없음 → OOB 체크 불필요
            terminated = goal_reached | collision
        else:
            # 벽 없으면 map 이탈도 종료 조건
            local_pos  = robot_pos - self.scene.env_origins[:, :2]
            out_of_bounds = (
                (local_pos[:, 0].abs() > self.cfg.map_size) |
                (local_pos[:, 1].abs() > self.cfg.map_size)
            )
            terminated = goal_reached | collision | out_of_bounds

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

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0] = origins[:, 0] + rx
        root_state[:, 1] = origins[:, 1] + ry
        root_state[:, 2] = origins[:, 2]
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

        # ── 장애물 위치·반경 랜덤화 ──────────────────────────────────────────
        obs_range = self.cfg.map_size - 0.5
        for i in range(self.cfg.num_obstacles):
            ox = torch.empty(n, device=self.device).uniform_(-obs_range, obs_range)
            oy = torch.empty(n, device=self.device).uniform_(-obs_range, obs_range)
            self._obstacle_pos[env_ids, i, 0] = origins[:, 0] + ox
            self._obstacle_pos[env_ids, i, 1] = origins[:, 1] + oy
            self._obstacle_radii[env_ids, i] = torch.empty(n, device=self.device).uniform_(
                self.cfg.obstacle_radius_min, self.cfg.obstacle_radius_max
            )

        # ── Phase E: 물리 장애물 위치 리셋 ────────────────────────────────────
        if self.cfg.use_physical_obstacles:
            for i, obs_obj in enumerate(self._obstacle_objects):
                new_state = obs_obj.data.default_root_state[env_ids].clone()
                new_state[:, 0] = self._obstacle_pos[env_ids, i, 0]
                new_state[:, 1] = self._obstacle_pos[env_ids, i, 1]
                new_state[:, 2] = origins[:, 2] + self.cfg.obstacle_height / 2.0
                # 속도 0
                new_state[:, 7:] = 0.0
                obs_obj.write_root_pose_to_sim(new_state[:, :7], env_ids)
                obs_obj.write_root_velocity_to_sim(new_state[:, 7:], env_ids)

        # ── 조향 rate limiter 초기화 ──────────────────────────────────────────
        self._prev_delta[env_ids] = 0.0

        # ── prev_goal_dist 초기화 ─────────────────────────────────────────────
        ddx = self._goal_pos[env_ids, 0] - self.robot.data.root_pos_w[env_ids, 0]
        ddy = self._goal_pos[env_ids, 1] - self.robot.data.root_pos_w[env_ids, 1]
        self._prev_goal_dist[env_ids] = torch.sqrt(ddx * ddx + ddy * ddy)

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼: 해석적 LiDAR
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_lidar(self) -> torch.Tensor:
        """80-sector 2D LiDAR 거리를 해석적으로 계산.

        - 장애물: ray-cylinder 교차 공식
        - 벽 (Phase E): ray-AABB 벽면 교차 공식

        Returns:
            Tensor (N, 80): 정규화된 LiDAR 거리 [0, 1].
        """
        N = self.num_envs
        S = self.cfg.num_sectors
        D = self.cfg.lidar_range

        robot_pos = self.robot.data.root_pos_w[:, :2]        # (N, 2)
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)  # (N,)

        # 각 섹터의 월드 프레임 방향 (단위 벡터)
        ray_angles = yaw.unsqueeze(1) + self._sector_offsets.unsqueeze(0)  # (N, S)
        ray_dx = torch.cos(ray_angles)   # (N, S)
        ray_dy = torch.sin(ray_angles)   # (N, S)

        # 거리 버퍼 초기화 (미감지 = max)
        distances = torch.full((N, S), D, device=self.device)

        # ── 장애물 ray-cylinder 교차 ──────────────────────────────────────────
        for i in range(self.cfg.num_obstacles):
            cx = (self._obstacle_pos[:, i, 0] - robot_pos[:, 0]).unsqueeze(1)  # (N,1)
            cy = (self._obstacle_pos[:, i, 1] - robot_pos[:, 1]).unsqueeze(1)

            t = cx * ray_dx + cy * ray_dy          # (N, S) — 중심 투영
            perp2 = cx * cx + cy * cy - t * t      # 수직 거리²

            r     = self._obstacle_radii[:, i].unsqueeze(1)  # (N,1)
            under = r * r - perp2
            hit   = (under >= 0) & (t > 0)

            cyl_dist = torch.where(
                hit,
                torch.clamp(t - torch.sqrt(torch.clamp(under, min=0.0)), min=0.0),
                torch.full_like(t, D),
            )
            distances = torch.minimum(distances, cyl_dist)

        # ── Phase E: 벽 ray-AABB 교차 ─────────────────────────────────────────
        if self.cfg.use_walls:
            m = self.cfg.map_size
            local_pos = robot_pos - self.scene.env_origins[:, :2]  # (N, 2)
            px = local_pos[:, 0:1]  # (N, 1)
            py = local_pos[:, 1:2]

            # 0으로 나누기 방지
            safe_dx = torch.where(ray_dx.abs() < 1e-8, torch.full_like(ray_dx, 1e-8), ray_dx)
            safe_dy = torch.where(ray_dy.abs() < 1e-8, torch.full_like(ray_dy, 1e-8), ray_dy)

            # 4개 벽면까지 거리 (양수인 것만 유효)
            t_xp = ( m - px) / safe_dx   # x = +m 벽
            t_xn = (-m - px) / safe_dx   # x = -m 벽
            t_yp = ( m - py) / safe_dy   # y = +m 벽
            t_yn = (-m - py) / safe_dy   # y = -m 벽

            # 음수(뒤쪽 교차) → max로 대체
            t_xp = torch.where(t_xp > 1e-4, t_xp, torch.full_like(t_xp, D))
            t_xn = torch.where(t_xn > 1e-4, t_xn, torch.full_like(t_xn, D))
            t_yp = torch.where(t_yp > 1e-4, t_yp, torch.full_like(t_yp, D))
            t_yn = torch.where(t_yn > 1e-4, t_yn, torch.full_like(t_yn, D))

            wall_dist = torch.minimum(
                torch.minimum(t_xp, t_xn),
                torch.minimum(t_yp, t_yn),
            )
            distances = torch.minimum(distances, wall_dist)

        # 정규화 [0, 1]
        return torch.clamp(distances, 0.0, D) / D

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼: 충돌 판정
    # ─────────────────────────────────────────────────────────────────────────

    def _get_collision_mask(self) -> torch.Tensor:
        """로봇이 장애물과 충돌했는지 (N,) bool.

        Phase E에서도 벽 물리 충돌은 여기서 판정하지 않음.
        (벽은 종료 조건에서 out_of_bounds로 처리)
        """
        robot_pos = self.robot.data.root_pos_w[:, :2]
        collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.cfg.num_obstacles):
            dist = torch.norm(robot_pos - self._obstacle_pos[:, i, :], dim=1)
            collision |= dist < (_ROBOT_RADIUS + self._obstacle_radii[:, i])
        return collision

    def _get_min_obstacle_dist(self) -> torch.Tensor:
        """로봇 표면에서 가장 가까운 장애물 표면까지 거리 (N,) [m]."""
        robot_pos = self.robot.data.root_pos_w[:, :2]
        min_dist  = torch.full((self.num_envs,), self.cfg.lidar_range, device=self.device)
        for i in range(self.cfg.num_obstacles):
            surface_dist = (
                torch.norm(robot_pos - self._obstacle_pos[:, i, :], dim=1)
                - _ROBOT_RADIUS
                - self._obstacle_radii[:, i]
            )
            min_dist = torch.minimum(min_dist, surface_dist)
        return min_dist.clamp(min=0.0)
