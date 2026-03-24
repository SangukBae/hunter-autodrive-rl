# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ouster OS1 LiDAR 모델 사양 정의 (Hunter SE V0 공용).

teleop_lidar_check.py  : RTX LiDAR (OmniLidar) 설정 참조
lidar_nav_env.py       : MultiMeshRayCaster channels 설정 참조

지원 모델:
    os1-32  : 32채널  / 10Hz / 2048res → 655,360 PPS   (Isaac Sim 내장 ✓)
    os1-64  : 64채널  / 10Hz / 2048res → 1,310,720 PPS (Isaac Sim 미내장 — RTX는 OS1-128 대체)
    os1-128 : 128채널 / 10Hz / 2048res → 2,621,440 PPS (Isaac Sim 내장 ✓)

OS1-64 RTX 참고:
    Isaac Sim 5.0 SUPPORTED_LIDAR_CONFIGS 에 OS1-64 variant 없음.
    RTX 렌더링 시 OS1-128 config 를 대체 사용 (128ch 점군 반환).
    학습 환경(MultiMeshRayCaster) 은 channels=64 를 그대로 사용 가능.
"""

from __future__ import annotations

# ── 모델 사양 테이블 ────────────────────────────────────────────────────────────
OS1_LIDAR_MODELS: dict[str, dict] = {
    "os1-32": {
        # Isaac Sim 5.0 내장 RTX config (SUPPORTED_LIDAR_CONFIGS 등록됨)
        "rtx_config": "OS1_REV6_32ch10hz2048res",
        # 물리 스펙
        "channels": 32,
        "pts_per_scan": 32 * 2048,           # 65,536 pts/scan (1 회전)
        "scan_rate_hz": 10,
        "azimuth_steps": 2048,
        "points_per_second": 655_360,
        "vertical_fov_range": (-22.5, 22.5),  # 45° V-FOV
        "max_distance_m": 120.0,
        "near_range_m": 0.3,
        "range_accuracy_m": 0.03,
    },
    "os1-64": {
        # Isaac Sim 5.0 에 OS1-64 내장 RTX config 없음 → OS1-128 대체 사용
        "rtx_config": "OS1_REV6_128ch10hz2048res",
        "rtx_note": "Isaac Sim 5.0 에 OS1-64 RTX config 미내장. OS1-128(128ch)로 대체 실행.",
        # 물리 스펙 (OS1-64 기준; RTX는 128ch 실제 렌더링)
        "channels": 64,
        "pts_per_scan": 128 * 2048,           # 262,144 (OS1-128 RTX 기준)
        "scan_rate_hz": 10,
        "azimuth_steps": 2048,
        "points_per_second": 1_310_720,        # 64ch 기준 PPS
        "vertical_fov_range": (-22.5, 22.5),
        "max_distance_m": 120.0,
        "near_range_m": 0.3,
        "range_accuracy_m": 0.03,
    },
    "os1-128": {
        # Isaac Sim 5.0 내장 RTX config
        "rtx_config": "OS1_REV6_128ch10hz2048res",
        # 물리 스펙
        "channels": 128,
        "pts_per_scan": 128 * 2048,           # 262,144 pts/scan (1 회전)
        "scan_rate_hz": 10,
        "azimuth_steps": 2048,
        "points_per_second": 2_621_440,
        "vertical_fov_range": (-22.5, 22.5),
        "max_distance_m": 120.0,
        "near_range_m": 0.3,
        "range_accuracy_m": 0.03,
    },
}

VALID_LIDAR_MODELS: list[str] = list(OS1_LIDAR_MODELS.keys())
DEFAULT_LIDAR_MODEL: str = "os1-32"


def get_lidar_spec(model: str) -> dict:
    """모델 이름으로 LiDAR 사양 딕셔너리를 반환한다.

    Args:
        model: 모델 이름. ``"os1-32"``, ``"os1-64"``, ``"os1-128"`` 중 하나.

    Returns:
        해당 모델의 사양 딕셔너리.

    Raises:
        ValueError: 알 수 없는 모델 이름이 전달된 경우.
    """
    if model not in OS1_LIDAR_MODELS:
        raise ValueError(
            f"알 수 없는 LiDAR 모델: {model!r}. 선택 가능: {VALID_LIDAR_MODELS}"
        )
    return OS1_LIDAR_MODELS[model]
