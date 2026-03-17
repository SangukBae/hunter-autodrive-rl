# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE 평탄 트랙 경로 추종 환경.

원본 출처:
    Hybrid_DRL_Deployments/.../hunter_hybrid/hunter_hybrid_env.py
    (리팩토링 및 모듈화)
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import add_reference_to_stage
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_autodrive.utils.cubic_spline import calc_spline_course

from .path_tracking_env_cfg import HunterPathTrackingEnvCfg


class HunterPathTrackingEnv(DirectRLEnv):
    """Hunter SE 경로 추종 환경.

    Isaac Lab DirectRLEnv 를 상속합니다.
    GPU 병렬 환경에서 4096개 에이전트를 동시에 학습합니다.

    관측 (7D):
        x, y, crosstrack_error, heading_error, roll, yaw, linear_velocity

    행동 (2D):
        velocity:       [-1, 1] → [0, 21.82] rad/s (후륜 각속도)
        steering_angle: [-1, 1] → [-0.384, 0.384] rad (중앙 조향각, 22°)

    Ackermann 조향 변환:
        delta_out = atan(L * tan(delta) / (L + 0.5 * tw * tan(delta)))
        delta_in  = atan(L * tan(delta) / (L - 0.5 * tw * tan(delta)))
    """

    cfg: HunterPathTrackingEnvCfg

    # Hunter SE 제원
    WHEELBASE: float = 0.550   # 축거 [m]  — PDF: Front/rear track 550 mm
    TRACK_WIDTH: float = 0.460 # 윤거 [m]  — PDF: Axle Track 460 mm
    MAX_STEER: float = 0.384   # 최대 조향각 [rad] (22°) — PDF: Max inner wheel steering angle 22°
    MAX_WHEEL_VEL: float = 21.82  # 최대 휠 각속도 [rad/s]

    def __init__(self, cfg: HunterPathTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 조인트 인덱스 캐싱
        self._lw_idx, _ = self.hunter.find_joints("re_left_jiont")
        self._rw_idx, _ = self.hunter.find_joints("re_right_jiont")
        self._fsl_idx, _ = self.hunter.find_joints("fr_steer_left_joint")
        self._fsr_idx, _ = self.hunter.find_joints("fr_steer_right_joint")

        self.joint_pos = self.hunter.data.joint_pos
        self.joint_vel = self.hunter.data.joint_vel

        # 리셋 플래그 (보상 페널티용)
        self.hunter_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # 씬 설정
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        """씬 초기화: 로봇 스폰, 경로 생성, 환경 복제."""
        self.hunter = Articulation(self.cfg.robot)

        # 트랙 CSV 로드 및 스플라인 생성
        coords = np.genfromtxt(self.cfg.track_csv, delimiter=",")
        x_coords = coords[:: self.cfg.waypoint_downsample, 0]
        y_coords = coords[:: self.cfg.waypoint_downsample, 1]

        self.cx, self.cy, self.cyaw, self.ck, _ = calc_spline_course(
            x_coords, y_coords, ds=self.cfg.spline_ds
        )

        # 스플라인 좌표를 각 환경에 오프셋 적용하여 텐서로 변환
        coordinates_tensor = torch.tensor(
            np.stack((self.cx, self.cy), axis=-1), dtype=torch.float32
        )
        self.cyaw_torch = torch.tensor(self.cyaw, dtype=torch.float32)

        num_per_row = int(math.sqrt(self.num_envs))
        num_rows = math.ceil(self.num_envs / num_per_row)
        num_cols = math.ceil(self.num_envs / num_rows)
        spacing = self.cfg.scene.env_spacing

        row_offset = 0.5 * spacing * (num_rows - 1)
        col_offset = 0.5 * spacing * (num_cols - 1)

        translations = []
        for i in range(self.num_envs):
            row = i // num_cols
            col = i % num_cols
            translations.append([row_offset - row * spacing, col * spacing - col_offset])

        translations_tensor = torch.tensor(translations, dtype=torch.float32)

        # shape: (num_envs, num_path_points, 2)
        self.translated_coordinates = (
            coordinates_tensor.unsqueeze(0) + translations_tensor.unsqueeze(1)
        )

        # 지형 스폰: terrain_usd가 있으면 USD 지형, 없으면 기본 평지
        if self.cfg.terrain_usd is not None:
            add_reference_to_stage(usd_path=self.cfg.terrain_usd, path="/World/Terrain")
        else:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 환경 복제 및 충돌 필터링
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["hunter"] = self.hunter

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 트랙 경로 시각화
        self._draw_track_visualization()

    def _draw_track_visualization(self) -> None:
        """트랙 경로를 debug_draw로 시각화합니다.

        env_spacing 기준으로 최대 8개 환경에 노란색 중심선을 그립니다.
        """
        try:
            from omni.isaac.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            return

        max_vis = min(self.num_envs, 8)
        z = 0.1  # 지면 위 높이 [m]

        for env_idx in range(max_vis):
            tx = self.translated_coordinates[env_idx, :, 0].cpu().numpy().tolist()
            ty = self.translated_coordinates[env_idx, :, 1].cpu().numpy().tolist()
            n = len(tx)

            # 연속된 선분으로 트랙 그리기 (노란색, 두께 3)
            point_list_0 = [(tx[i],     ty[i],     z) for i in range(n - 1)]
            point_list_1 = [(tx[i + 1], ty[i + 1], z) for i in range(n - 1)]
            colors  = [(1.0, 1.0, 0.0, 1.0)] * (n - 1)  # RGBA 노란색
            sizes   = [3.0] * (n - 1)

            draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    # ─────────────────────────────────────────────────────────────────────────
    # 물리 스텝 전처리 (행동 스케일링)
    # ─────────────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """정규화된 행동 [-1, 1] 을 실제 제어값으로 변환."""
        self.actions = actions.clone()

        # velocity: [-1, 1] → [0, MAX_WHEEL_VEL]
        self.actions[:, 0] = (actions[:, 0] + 1.0) / 2.0 * self.MAX_WHEEL_VEL

        # steering: [-1, 1] → [-MAX_STEER, MAX_STEER]
        self.actions[:, 1] = actions[:, 1] * self.MAX_STEER

    # ─────────────────────────────────────────────────────────────────────────
    # 행동 적용 (Ackermann 조향 변환)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self) -> None:
        """Ackermann 기하학으로 좌/우 조향각 계산 후 관절 목표값 적용."""
        delta = self.actions[:, 1]
        L = self.WHEELBASE
        tw = self.TRACK_WIDTH

        tan_delta = torch.tan(delta)

        delta_out = torch.atan(L * tan_delta / (L + 0.5 * tw * tan_delta))
        delta_in  = torch.atan(L * tan_delta / (L - 0.5 * tw * tan_delta))

        # 조향 방향에 따라 내/외륜 구분
        steer_right = torch.where(delta <= 0, delta_in,  delta_out)
        steer_left  = torch.where(delta >  0, delta_in,  delta_out)

        wheel_vel = self.actions[:, 0]

        self.hunter.set_joint_position_target(steer_right.unsqueeze(-1), joint_ids=self._fsr_idx)
        self.hunter.set_joint_position_target(steer_left.unsqueeze(-1),  joint_ids=self._fsl_idx)
        self.hunter.set_joint_velocity_target(wheel_vel.unsqueeze(-1),   joint_ids=self._lw_idx)
        self.hunter.set_joint_velocity_target(wheel_vel.unsqueeze(-1),   joint_ids=self._rw_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # 관측 계산
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """7D 관측 벡터 반환."""
        self.translated_coordinates = self.translated_coordinates.to(self.device)
        self.cyaw_torch = self.cyaw_torch.to(self.device)

        pos_xy = self.hunter.data.root_pos_w[:, 0:2].unsqueeze(1)
        distances = torch.sqrt(torch.sum((pos_xy - self.translated_coordinates) ** 2, dim=-1))
        self.min_dist_idx = torch.argmin(distances, dim=1)

        # 횡방향 오차 및 헤딩 오차
        self.heading_w = self.hunter.data.heading_w
        cte = distances[torch.arange(self.num_envs), self.min_dist_idx]
        he = self.heading_w - self.cyaw_torch[self.min_dist_idx]
        he = (he + torch.pi) % (2 * torch.pi) - torch.pi

        # 부호 있는 횡방향 오차
        self.crosstrack_error = torch.where(he <= 0.0, -cte, cte)
        self.heading_error = he

        quat = self.hunter.data.root_quat_w
        roll, _, yaw = euler_xyz_from_quat(quat)

        obs = torch.cat(
            [
                self.hunter.data.root_pos_w[:, 0:2],
                self.crosstrack_error.unsqueeze(-1),
                self.heading_error.unsqueeze(-1),
                roll.unsqueeze(-1),
                yaw.unsqueeze(-1),
                self.hunter.data.root_lin_vel_b[:, 0].unsqueeze(-1),
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ─────────────────────────────────────────────────────────────────────────
    # 보상 계산
    # ─────────────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """보상 = exp(-cte/5) * exp(-|he|/π) * (0.1 * vel/3)."""
        cte_norm = torch.abs(self.crosstrack_error) / self.cfg.max_crosstrack_error
        he_norm  = torch.abs(self.heading_error) / math.pi
        vel      = torch.abs(self.hunter.data.root_lin_vel_b[:, 0]) / 3.0

        reward = torch.exp(-cte_norm) * torch.exp(-he_norm) * (0.1 * vel)

        # 리셋 페널티
        reward = reward - 1.0 * self.hunter_reset.float()
        return torch.clamp(reward, min=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 종료 조건
    # ─────────────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료: 오차 초과 또는 정지, 시간 초과."""
        pos_xy = self.hunter.data.root_pos_w[:, 0:2].unsqueeze(1)
        distances = torch.sqrt(torch.sum((pos_xy - self.translated_coordinates) ** 2, dim=-1))
        min_idx = torch.argmin(distances, dim=1)
        cte = distances[torch.arange(self.num_envs), min_idx]

        vel = self.hunter.data.root_lin_vel_b[:, 0]

        grace = self.episode_length_buf >= self.cfg.velocity_check_start_step
        terminated = (cte >= self.cfg.max_crosstrack_error) | (grace & (vel <= self.cfg.min_velocity))
        time_out   = self.episode_length_buf >= self.max_episode_length - 1

        self.hunter_reset = terminated
        return terminated, time_out

    # ─────────────────────────────────────────────────────────────────────────
    # 에피소드 리셋
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """지정된 환경들을 초기 상태로 리셋."""
        if env_ids is None:
            env_ids = self.hunter._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.hunter.data.default_joint_pos[env_ids]
        joint_vel = self.hunter.data.default_joint_vel[env_ids]

        default_root_state = self.hunter.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.hunter.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.hunter.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.hunter.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.hunter_reset[env_ids] = False
