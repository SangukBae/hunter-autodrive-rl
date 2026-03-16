# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TensorBoard + JSON 로거.

사용 예시:
    logger = Logger("logs/tqc/run1", use_tensorboard=True)
    logger.log_scalar("train/reward", 1.23, step=100)
    logger.close()
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


class Logger:
    """스칼라 값을 TensorBoard와 JSON 파일에 동시 기록.

    Args:
        log_dir:          로그 저장 디렉토리
        use_tensorboard:  TensorBoard SummaryWriter 사용 여부
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._data: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._json_path = os.path.join(log_dir, "metrics.json")

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[Logger] TensorBoard 미설치 — JSON 로깅만 사용합니다.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """스칼라 값을 TensorBoard와 JSON에 기록.

        Args:
            tag:   메트릭 이름 (예: "train/reward")
            value: 기록할 값
            step:  글로벌 스텝
        """
        self._data[tag].append((step, float(value)))
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """딕셔너리의 모든 항목을 일괄 기록."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def flush(self) -> None:
        """JSON 파일로 즉시 저장."""
        with open(self._json_path, "w") as f:
            json.dump(self._data, f, indent=2)
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """로거 종료 및 JSON 저장."""
        self.flush()
        if self.writer is not None:
            self.writer.close()
        print(f"[Logger] 메트릭 저장 완료: {self._json_path}")
