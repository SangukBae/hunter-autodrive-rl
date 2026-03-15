# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""레이싱 트랙 지형 설정."""

import os

# 트랙 CSV 파일 경로 상수
WAYPOINTS_DIR = os.path.join(os.path.dirname(__file__), "waypoints")

AUSTIN_CSV       = os.path.join(WAYPOINTS_DIR, "austin_centerline2.csv")
BRANDSHATCH_CSV  = os.path.join(WAYPOINTS_DIR, "brandshatch_centerline.csv")
SILVERSTONE_CSV  = os.path.join(WAYPOINTS_DIR, "silverstone_centerline.csv")

TRACK_CSV_MAP = {
    "austin":      AUSTIN_CSV,
    "brandshatch": BRANDSHATCH_CSV,
    "silverstone": SILVERSTONE_CSV,
}
