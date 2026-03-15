# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LAP (Latent Action Priority) Prioritized Experience Replay 버퍼.

원본 출처:
    drl_agent/scripts/utils/buffer.py
"""

from __future__ import annotations

import numpy as np
import torch


class LAP:
    """우선순위 경험 리플레이 버퍼.

    Args:
        state_dim: 상태 차원
        action_dim: 행동 차원
        device: PyTorch 디바이스
        max_size: 최대 버퍼 크기 (기본값: 1,000,000)
        batch_size: 샘플 배치 크기 (기본값: 256)
        max_action: 행동 최대값 (정규화용, 기본값: 1.0)
        normalize_actions: 행동 정규화 여부 (기본값: True)
        prioritized: 우선순위 샘플링 활성화 (기본값: False)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        max_size: int = 1_000_000,
        batch_size: int = 256,
        max_action: float = 1.0,
        normalize_actions: bool = True,
        prioritized: bool = False,
    ):
        self.max_size = int(max_size)
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state      = np.zeros((self.max_size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim),  dtype=np.float32)
        self.reward     = np.zeros((self.max_size, 1),          dtype=np.float32)
        self.not_done   = np.zeros((self.max_size, 1),          dtype=np.float32)

        self.prioritized = prioritized
        if prioritized:
            self.priority    = torch.zeros(self.max_size, device=device)
            self.max_priority = 1.0

        self.normalize_actions = max_action if normalize_actions else 1.0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: float,
    ) -> None:
        """트랜지션 하나를 버퍼에 추가."""
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr]     = reward
        self.not_done[self.ptr]   = 1.0 - done

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> tuple[torch.Tensor, ...]:
        """배치 샘플링. (state, action, next_state, reward, not_done) 반환."""
        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], dim=0)
            val  = torch.rand(self.batch_size, device=self.device) * csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=self.batch_size)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr[self.ind], dtype=torch.float32, device=self.device)

        return _t(self.state), _t(self.action), _t(self.next_state), _t(self.reward), _t(self.not_done)

    def update_priority(self, priority: torch.Tensor) -> None:
        """샘플된 인덱스의 우선순위 업데이트."""
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self) -> None:
        """최대 우선순위 재계산."""
        self.max_priority = float(self.priority[: self.size].max())

    def __len__(self) -> int:
        return self.size
