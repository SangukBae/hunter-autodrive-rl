# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR 자율주행 환경 설정.

관측 공간 (82D):
    [lidar_0 ~ lidar_79 (80 sector), goal_dist, goal_angle]

행동 공간 (2D):
    [linear_vel (-1 ~ 1), angular_vel (-1 ~ 1)]

보상:
    goal_progress (dense) + goal_reached (sparse +100)
    + collision (sparse -10) + obstacle_proximity (smooth)
    + time_penalty (-0.01/step)

종료 조건:
    - goal_dist < goal_threshold       → 목표 도달
    - 장애물 충돌 (proximity < ROBOT_RADIUS + obs_radius)
    - episode 시간 초과
    - map 이탈 (use_walls=False 일 때만 유효)

Phase 구분:
    Phase C  : use_walls=False, use_physical_obstacles=False (기본)
    Phase E  : use_walls=True,  use_physical_obstacles=True
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_autodrive.assets.robots.hunter import HUNTER_SE_CFG


@configclass
class LidarNavEnvCfg(DirectRLEnvCfg):
    """LiDAR 자율주행 환경 설정 (학습용)."""

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,        # STEP_DT = 0.02s (drive_hunter_se.py와 동일)
        use_fabric=True,
        enable_scene_query_support=True,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            enable_external_forces_every_iteration=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
        ),
    )

    # ── 로봇 ────────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = HUNTER_SE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ── 씬 ──────────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=20.0,   # 맵(16×16)보다 여유 있게
        replicate_physics=True,
    )

    # ── 환경 파라미터 ───────────────────────────────────────────────────────────
    decimation: int = 4                  # 제어 주기 = dt × decimation = 0.02s
    episode_length_s: float = 60.0
    action_space: int = 2                # [linear_vel, angular_vel]
    observation_space: int = 82          # 80 LiDAR sectors + [goal_dist, goal_angle]
    state_space: int = 0

    # ── LiDAR ───────────────────────────────────────────────────────────────────
    lidar_range: float = 5.0             # 최대 감지 거리 [m]
    num_sectors: int = 80                # 360° / 80 = 4.5° per sector

    # ── 맵 / 장애물 ─────────────────────────────────────────────────────────────
    map_size: float = 8.0                # ±8m → 16×16 m 맵
    num_obstacles: int = 5
    obstacle_radius_min: float = 0.15    # [m]
    obstacle_radius_max: float = 0.5     # [m]
    obstacle_height: float = 1.2         # [m]
    min_spawn_dist: float = 1.5          # 로봇/goal에서 최소 이격 거리 [m]

    # ── 목표 ────────────────────────────────────────────────────────────────────
    goal_threshold: float = 0.3          # 목표 도달 판정 거리 [m]

    # ── 보상 파라미터 ───────────────────────────────────────────────────────────
    goal_reward: float = 100.0
    collision_penalty: float = -10.0
    proximity_threshold: float = 0.8     # 이 거리 이내부터 근접 페널티 [m]
    proximity_penalty_max: float = 2.0   # 근접 페널티 최대값
    progress_scale: float = 5.0          # goal_progress 보상 스케일
    time_penalty: float = -0.01          # 스텝당 생존 비용

    # ── 액션 스케일 ─────────────────────────────────────────────────────────────
    max_linear_vel: float = 1.0          # [m/s]
    max_angular_vel: float = 1.0         # [rad/s]

    # ── Phase E 파라미터 (기본 비활성) ───────────────────────────────────────────
    use_walls: bool = False              # True → 4개 물리 벽 스폰
    wall_thickness: float = 0.4         # [m]
    wall_height: float = 1.5            # [m]
    use_physical_obstacles: bool = False # True → 물리 장애물 실린더 스폰


@configclass
class LidarNavEnvCfgPlay(LidarNavEnvCfg):
    """시각화/테스트용 환경 설정 (소규모, Phase C)."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=20.0,
        replicate_physics=True,
    )


@configclass
class LidarNavEnvCfgPhaseE(LidarNavEnvCfg):
    """Phase E: 물리 벽 + 물리 장애물 + 완전한 16×16 맵.

    장애물 반경을 단일값(0.3m)으로 고정하여 물리 메시와 일치시킵니다.
    LiDAR는 장애물(원통 해석적) + 벽(평면 해석적)을 함께 감지합니다.
    """

    # 물리 벽 + 물리 장애물 활성화
    use_walls: bool = True
    use_physical_obstacles: bool = True

    # 물리 메시와 일치시키기 위해 단일 반경 사용
    obstacle_radius_min: float = 0.3
    obstacle_radius_max: float = 0.3


@configclass
class LidarNavEnvCfgPhaseEPlay(LidarNavEnvCfgPhaseE):
    """Phase E 시각화용 소규모 환경."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=20.0,
        replicate_physics=True,
    )
