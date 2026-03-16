# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 평탄 트랙 경로 추종 환경 설정.

관측 공간 (7D):
    [x, y, crosstrack_error, heading_error, roll, yaw, linear_velocity]

행동 공간 (2D):
    [velocity (-1 ~ 1 → 0 ~ 21.82 rad/s), steering_angle (-0.524 ~ 0.524 rad)]

보상:
    reward = exp(-cte/5.0) * exp(-he/pi) * (0.1 * vel/3.0)

종료 조건:
    - crosstrack_error >= 5.0 m
    - linear_velocity <= 0.01 m/s
    - 에피소드 시간 초과 (200s)
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_autodrive.assets.robots.hunter import HUNTER_CFG
from isaaclab_autodrive.assets.terrains import ROUGH_TERRAIN_USD
from isaaclab_autodrive.terrains.track import AUSTIN_CSV


@configclass
class HunterPathTrackingEnvCfg(DirectRLEnvCfg):
    """평탄 트랙 경로 추종 환경 설정 (학습용)."""

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,           # 물리 스텝: 0.005s
        render_interval=20,   # 20 스텝마다 렌더링 → 제어 주기 0.1s
        use_fabric=True,
        enable_scene_query_support=False,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=1,               # 0 → 1: 속도 계산 정확도 향상
            enable_external_forces_every_iteration=True,  # 속도 노이즈 감소
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=5 * 2**15,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_heap_capacity=2**26,
            gpu_temp_buffer_capacity=2**24,
            gpu_max_num_partitions=8,
        ),
    )

    # ── 로봇 ────────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = HUNTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ── 씬 ──────────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # ── 환경 파라미터 ───────────────────────────────────────────────────────────
    decimation: int = 20              # 제어 주기: dt * decimation = 0.1s
    episode_length_s: float = 200.0   # 에피소드 길이 [s]
    action_space: int = 2             # [velocity, steering_angle]
    observation_space: int = 7        # [x, y, cte, he, roll, yaw, vel]
    state_space: int = 0

    # ── 지형 설정 ───────────────────────────────────────────────────────────────
    terrain_usd: str | None = ROUGH_TERRAIN_USD  # 지형 USD 경로 (None이면 기본 평지)

    # ── 트랙 설정 ───────────────────────────────────────────────────────────────
    track_csv: str = AUSTIN_CSV       # 학습에 사용할 트랙 CSV 경로
    waypoint_downsample: int = 10     # CSV 에서 10개 간격으로 샘플링
    spline_ds: float = 0.1            # 스플라인 보간 간격 [m]

    # ── 리셋 조건 ───────────────────────────────────────────────────────────────
    max_crosstrack_error: float = 5.0   # 최대 허용 횡방향 오차 [m]
    min_velocity: float = 0.01          # 최소 속도 (정지 감지) [m/s]


@configclass
class HunterPathTrackingEnvCfgPlay(HunterPathTrackingEnvCfg):
    """시각화/테스트용 환경 설정 (소규모)."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=5.0,
        replicate_physics=True,
    )
