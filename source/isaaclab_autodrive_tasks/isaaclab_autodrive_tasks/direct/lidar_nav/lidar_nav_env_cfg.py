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
    - LiDAR 최소 거리 < lidar_collision_threshold → 충돌
    - episode 시간 초과

맵:
    16×16 m 벽으로 둘러싸인 정사각형, 매 에피소드 10개 장애물 랜덤 배치
    로봇 스폰·목표점 위치 매 에피소드 랜덤
    RTX LiDAR (OS1-32) 기반 포인트 클라우드 학습
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_autodrive.assets.robots.hunter import HUNTER_SE_V0_CFG


@configclass
class LidarNavEnvCfg(DirectRLEnvCfg):
    """LiDAR 자율주행 환경 설정 (학습용)."""

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,        # STEP_DT = 0.02s
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

    # ── 로봇 (Hunter SE V0 — 공식 메뉴얼 스펙 기반, 물리 파라미터 검증 완료) ──
    robot: ArticulationCfg = HUNTER_SE_V0_CFG.replace(
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
    lidar_model: str = "os1-32"          # RTX OmniLidar 모델
    lidar_use_rtx: bool = True           # RTX OmniLidar (실물 OS1 동일 방식)
    lidar_range: float = 120.0           # 최대 감지 거리 [m] — OS1 실물 스펙
    num_sectors: int = 80                # 360° / 80 = 4.5° per sector

    # ── 맵 / 장애물 ─────────────────────────────────────────────────────────────
    map_size: float = 8.0                # ±8m → 16×16 m 맵
    num_obstacles: int = 10              # 매 에피소드 랜덤 배치 장애물 수
    obstacle_radius_min: float = 0.15    # [m]
    obstacle_radius_max: float = 0.5     # [m]
    obstacle_height: float = 1.2         # [m]
    min_spawn_dist: float = 1.5          # 로봇/goal에서 최소 이격 거리 [m]

    # ── 충돌 감지 (LiDAR 기반) ───────────────────────────────────────────────────
    lidar_collision_threshold: float = 0.3  # LiDAR 최소 거리 < 이 값 → 충돌 판정 [m]

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

    # ── 맵 구성 ─────────────────────────────────────────────────────────────────
    use_walls: bool = True               # 16×16 맵 경계 물리 벽
    wall_thickness: float = 0.4         # [m]
    wall_height: float = 1.5            # [m]
    use_physical_obstacles: bool = True  # 물리 장애물 실린더 스폰

    # ── 시각화 ───────────────────────────────────────────────────────────────
    lidar_debug_vis: bool = False        # True → 뷰포트에 ray hit 포인트 표시


@configclass
class LidarNavEnvCfgPlay(LidarNavEnvCfg):
    """시각화/테스트용 환경 설정 (소규모)."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=20.0,
        replicate_physics=True,
    )
    lidar_debug_vis: bool = True
