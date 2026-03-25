# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hunter SE V0 로봇을 위한 LiDAR 섹터 기반 자율주행 RL 학습 프레임워크.
NVIDIA Isaac Lab 위에서 오프-폴리시 알고리즘으로 목표 도달 + 장애물 회피 정책을 학습한다.

- **로봇**: Hunter SE V0 (`hunter_se_v0/`)
- **LiDAR 입력**: 포인트클라우드를 N개 섹터로 분할 → 섹터별 최솟값 벡터를 RL 입력으로 직접 사용
- **시뮬레이터**: NVIDIA Isaac Lab (Isaac Sim 5.0 기반)

## Commands

모든 스크립트는 Isaac Lab 런처를 통해 실행한다:

```bash
# 환경 패키지 설치 (최초 1회 또는 코드 수정 후)
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/env

# USD 파일 재생성 (로봇 구조 변경 시)
/workspace/isaaclab/isaaclab.sh -p hunter_se_v0/build_usd.py
```

## Repository Structure

```
hunter_se_v0/          # 로봇 에셋 패키지 (변경 빈도 낮음)
envs/lidar_nav/        # Isaac Lab DirectRLEnv 구현 (환경 핵심)
rl/
  agents/              # SAC, TQC 등 에이전트
  networks/            # Actor, Critic 네트워크
  buffers/             # Replay buffer
configs/
  env/                 # 환경 YAML 하이퍼파라미터
  rl/                  # 알고리즘 YAML 하이퍼파라미터
scripts/               # train.py, play.py 등 진입점 (로직 없음)
logs/                  # 학습 결과 (gitignored)
```

## hunter_se_v0 패키지

로봇 에셋 패키지. Isaac Sim Python 환경에서만 import 가능하다.

- **`hunter_se_v0_cfg.py`** — `HUNTER_SE_V0_CFG` (`ArticulationCfg`). 환경에서 `import`해서 사용.
  - USD 파일이 없으면 `build_usd.py`를 자동 호출해 생성한다.
- **`ackermann.py`** — `HunterSEAckermann` 클래스. RL action(선속도, 조향각) → 관절 목표값 변환.
  - `compute(lin_vel, delta_c)`: 배치 텐서 입력 (N,) → (steer_l, steer_r, omega_l, omega_r)
- **`lidar_cfg.py`** — Ouster OS1 LiDAR 모델 사양 테이블 (`get_lidar_spec(model)`).
- **`build_usd.py`** — pxr Python API로 USD articulation을 동적 생성. STL 메시는 `/robot_isaac/ugv_gazebo_sim/` 경로 참조.

### 물리 상수 (Hunter SE V0)

```
WHEELBASE    = 0.548 m
FRONT_TRACK  = 0.492 m
REAR_TRACK   = 0.504 m
WHEEL_RADIUS = 0.1375 m
MAX_STEER    = 0.384 rad (±22°)
MAX_SPEED    = 1.333 m/s (4.8 km/h)
spawn_z      = 0.2955 m  (바퀴 접지 높이)
```

### 액추에이터 설계

- 후륜 구동: `DCMotorCfg` — velocity mode, damping=15
- 전륜 조향: `ImplicitActuatorCfg` — position mode, stiffness=500, damping=50
- 전륜 자유회전: `ImplicitActuatorCfg` — stiffness=0, damping=0.01

### USD 계층 구조

플랫 계층 (모든 링크가 루트의 직접 자식) — PhysX RigidBody 중첩 방지:
```
/HunterSEV0  (ArticulationRootAPI)
  /base_link, /fr_steer_left_link, /fr_left_link,
  /fr_steer_right_link, /fr_right_link, /re_left_link, /re_right_link
  /Physics/  (관절 Scope)
```
조향 관절: `axis=Z` / 바퀴 관절: `axis=Y` (양수 속도 = 전진)

## Isaac Lab 환경 개발 시 주의사항

- `clone_environments(copy_from_source=False)` 사용 시 env_1+ 가 env_0의 라이브 미러가 됨. 복제 이후 prim 스폰은 env_0 로컬 좌표 기준으로만 수행 (루프 금지).
- RTX LiDAR는 환경당 독립 render product가 필요하므로 num_envs 4~8 권장.
- Isaac Sim 5.0 기준 지원 RTX LiDAR config: `OS1_REV6_32ch10hz2048res`, `OS1_REV6_128ch10hz2048res` (OS1-64 미지원 → OS1-128 대체).
