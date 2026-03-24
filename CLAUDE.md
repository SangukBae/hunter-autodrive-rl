# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hunter SE V0 로봇을 위한 LiDAR 기반 자율주행 RL 학습 프레임워크. NVIDIA Isaac Lab 위에서 TQC/TD7 오프-폴리시 알고리즘으로 목표 도달 + 장애물 회피 정책을 학습한다.

- **로봇**: Hunter SE V0 (`hunter_se_v0/`)
- **맵**: 16×16m 물리 벽 경계, 매 에피소드 10개 장애물 랜덤 배치
- **LiDAR**: RTX OmniLidar (OS1-32, 물리 기반 광자 추적 — 실물 동일 방식)
- **충돌 판정**: LiDAR 최소 감지 거리 기반

## Commands

모든 스크립트는 Isaac Lab 런처를 통해 실행한다:

```bash
# 패키지 설치 (최초 1회)
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks

# TQC 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 --algo tqc --num_envs 4 --headless

# TD7 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 --algo td7 --num_envs 4 --headless

# 정책 시각화
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-Play-v0 --algo tqc \
    --checkpoint logs/tqc/hunter_tqc/TIMESTAMP/model_final.pt --num_envs 4

# 환경 단위 테스트
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/test_lidar_nav.py

# LiDAR 텔레오프 체크
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_lidar_check.py
```

> RTX LiDAR는 환경당 독립 render product를 생성하므로 num_envs 4~8 권장.

## Architecture

### Package Structure

```
hunter_se_v0/                        # 로봇 에셋 (USD + ArticulationCfg + Ackermann)
source/
├── isaaclab_autodrive/              # 로봇 에셋 래퍼
│   └── assets/robots/hunter.py     # HUNTER_SE_V0_CFG 재익스포트
└── isaaclab_autodrive_tasks/        # Gymnasium 환경 + 등록
    └── direct/lidar_nav/
        ├── __init__.py              # 2개 환경 gym.register()
        ├── lidar_nav_env_cfg.py     # 환경 설정 dataclass
        └── lidar_nav_env.py         # LidarNavEnv 구현

scripts/autodrive/
├── train_tqc.py                     # 학습 진입점 (TQC/TD7 공용)
├── play_lidar_nav.py                # 정책 평가
├── test_lidar_nav.py                # 환경 단위 테스트
├── teleop_lidar_check.py            # LiDAR 동작 확인
└── algorithms/
    ├── tqc/                         # TQC 구현 (networks, agent, trainer)
    ├── td7/                         # TD7 구현 (SALE + LAP PER + ckpt)
    └── common/                      # LAP 버퍼, TensorBoard 로거
```

### 등록된 Gymnasium 환경

| 환경 ID | 용도 |
|---|---|
| `Isaac-LidarNav-Hunter-v0` | 학습용 (64 envs, RTX LiDAR) |
| `Isaac-LidarNav-Hunter-Play-v0` | 시각화 (4 envs) |

### LidarNavEnv 핵심 흐름

`DirectRLEnv`를 상속. 주요 메서드:

- `_setup_scene()`: 로봇, 평지, 조명, 물리 벽(4개), 물리 장애물 스폰
- `_pre_physics_step(actions)`: action → Ackermann 변환 → 조향 rate limiting → 관절 목표값 설정
- `_apply_action()`: `decimation` 횟수(4회)마다 물리 스텝에 관절 명령 전달
- `_compute_lidar()`: RTX OmniLidar 부분 스캔 → 롤링-min 버퍼 → 80-sector 정규화 맵
- `_get_observations()`: LiDAR obs 계산 + `_last_lidar_obs` 캐시 갱신
- `_get_collision_mask()`: `_last_lidar_obs.min() * lidar_range < lidar_collision_threshold`
- `_reset_idx(env_ids)`: 로봇/목표/장애물 랜덤 배치, RTX 버퍼 초기화

### RTX LiDAR 스캔 누적

OS1-32 @ 10Hz → 1회전 = 20 physics step ≈ 5 RL step.
각 step에서 수신된 부분 스캔(~72°)을 `_rtx_sector_buf`에 min으로 누적.
버퍼는 에피소드 리셋(`_reset_idx`) 시에만 `lidar_range`로 초기화됨.

### Action Space

```
act[0]: [-1, 1] → linear_vel [0, max_linear_vel m/s]  (전진 전용, 후진 없음)
act[1]: [-1, 1] → 차체 중심 조향각 delta [-0.384, +0.384 rad]
```

조향각 → Ackermann 요레이트: `ang_vel = lin_vel × tan(delta) / WHEELBASE`
조향 변화율 제한: `±0.5 rad/s` (`_prev_delta`로 추적)

### 로봇 물리 상수 (Hunter SE V0)

```python
_WHEELBASE    = 0.548 m
_REAR_TRACK   = 0.504 m
_WHEEL_RADIUS = 0.1375 m
_MAX_STEER    = 0.384 rad  # ±22°
spawn_z       = 0.2955 m   # 바퀴 접지 높이 (init_state.pos[2])
```

액추에이터: 후륜 `DCMotorCfg damping=15` (velocity mode), 조향 `stiffness=1e7, damping=1e5` (position mode)

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
