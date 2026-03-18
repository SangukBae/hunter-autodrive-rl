# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hunter SE 로봇을 위한 LiDAR 기반 자율주행 RL 학습 프레임워크. Isaac Lab (NVIDIA Isaac Sim) 위에서 TQC/TD7 오프-폴리시 알고리즘으로 목표 도달 + 장애물 회피 정책을 학습한다.

## Commands

모든 스크립트는 Isaac Lab 런처를 통해 실행한다:

```bash
# 패키지 설치 (최초 1회)
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks

# TQC 학습 (Phase C — 해석적 LiDAR, 가상 장애물)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 --algo tqc --num_envs 64 --headless

# TD7 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 --algo td7 --num_envs 64 --headless

# Phase E 학습 (물리 벽 + 물리 장애물)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-PhaseE-v0 --algo tqc --num_envs 64 --headless

# 정책 시각화
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-Play-v0 --algo tqc \
    --checkpoint logs/tqc/hunter_tqc/TIMESTAMP/model_final.pt --num_envs 4

# 환경 단위 테스트
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/test_lidar_nav.py

# Ackermann 주행 검증 (물리 파라미터 확인용)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_hunter_se.py
```

## Architecture

### Package Structure

```
source/
├── isaaclab_autodrive/          # 로봇 에셋 + 유틸리티
│   └── assets/robots/hunter.py  # HUNTER_SE_CFG (hunter_se/를 참조)
└── isaaclab_autodrive_tasks/    # Gymnasium 환경 + 등록
    └── direct/lidar_nav/        # 메인 태스크
        ├── __init__.py          # 4개 환경 gym.register()
        ├── lidar_nav_env_cfg.py # 환경 설정 dataclass
        └── lidar_nav_env.py     # LidarNavEnv 구현

hunter_se/
├── hunter_se_description.usda  # URDF→USD 변환 로봇 에셋
└── hunter_se_cfg.py             # ArticulationCfg (원본, 검증 완료)

scripts/autodrive/
├── train_tqc.py                 # 학습 진입점 (TQC/TD7 공용)
├── play_lidar_nav.py            # 정책 평가
└── algorithms/
    ├── tqc/                     # TQC 구현 (networks, agent, trainer)
    ├── td7/                     # TD7 구현 (SALE + LAP PER + ckpt)
    └── common/                  # LAP 버퍼, TensorBoard 로거
```

### 등록된 Gymnasium 환경

| 환경 ID | 용도 | 벽 | 물리 장애물 |
|---|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | 학습 (Phase C) | ✗ | ✗ |
| `Isaac-LidarNav-Hunter-Play-v0` | 시각화 | ✗ | ✗ |
| `Isaac-LidarNav-Hunter-PhaseE-v0` | 학습 (Phase E) | ✓ | ✓ |
| `Isaac-LidarNav-Hunter-PhaseE-Play-v0` | 시각화 | ✓ | ✓ |

### LidarNavEnv 핵심 흐름

`DirectRLEnv`를 상속. 주요 메서드:

- `_setup_scene()`: 로봇, 평지, 조명 스폰. `clone_environments(copy_from_source=False)` 호출 이후 Phase E 벽/장애물 추가
- `_pre_physics_step(actions)`: action → Ackermann 변환 → 조향 rate limiting → 관절 목표값 설정
- `_apply_action()`: `decimation` 횟수(4회)마다 물리 스텝에 관절 명령 전달
- `_compute_lidar()`: GPU 벡터화 해석적 LiDAR (Phase C: ray-cylinder, Phase E: + ray-AABB)
- `_reset_idx(env_ids)`: 로봇/목표/장애물 랜덤 배치, `_prev_delta` 초기화

### Action Space

```
act[0]: [-1, 1] → linear_vel [0, max_linear_vel m/s]  (전진 전용, 후진 없음)
act[1]: [-1, 1] → 차체 중심 조향각 delta [-0.384, +0.384 rad]
```

조향각 → Ackermann 요레이트: `ang_vel = lin_vel × tan(delta) / WHEELBASE`
조향 변화율 제한: `±0.5 rad/s` (물리적 타당성 확보, `_prev_delta`로 추적)

### 로봇 물리 상수

```python
_WHEELBASE    = 0.548 m
_REAR_TRACK   = 0.504 m
_WHEEL_RADIUS = 0.129 m
_MAX_STEER    = 0.384 rad  # ±22°
```

액추에이터: 후륜 `damping=17453` (velocity mode), 조향 `stiffness=1e7, damping=1e5` (position mode)

### IsaacLab 환경 복제 주의사항

`clone_environments(copy_from_source=False)`를 사용하면 env_1+ 가 env_0의 **라이브 미러**가 된다. env_0에 prim을 스폰하면 모든 env에 자동 반영되므로, 복제 이후 스폰 시 env_0에만 **로컬 좌표**로 스폰해야 한다 (루프 금지).

### 학습 루프 (TQCTrainer / TD7Trainer)

```
warmup_steps 동안: 랜덤 액션
이후:
  1. agent.select_action(obs) → action
  2. env.step(action) → (obs', reward, done)
  3. buffer.add() (num_envs개 전환 동시 저장)
  4. agent.train(batch) (updates_per_step회)
  5. eval_interval마다: 결정론적 정책으로 평가 + 체크포인트 저장
```

로그 출력: `logs/{algo}/{experiment_name}/{timestamp}/`

### YAML 하이퍼파라미터

각 태스크 디렉터리의 `agents/{algo}_cfg.yaml`에 위치. `--cfg` 옵션으로 경로 오버라이드 가능. `--num_envs`, `--seed`는 CLI에서 직접 오버라이드.
