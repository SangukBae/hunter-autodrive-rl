# Hunter SE Autonomous Driving RL

Hunter SE 모바일 로봇을 대상으로 **Isaac Lab 기반 강화학습**으로 LiDAR 기반 자율주행(목표 도달 + 장애물 회피)을 학습하고 실로봇에 배포하는 프로젝트입니다.

RL 학습 환경은 공식 메뉴얼 스펙 기반으로 물리 파라미터가 검증된 **Hunter SE V0** 로봇 모델을 사용합니다.

---

## 폴더 구조

```
hunter_autodrive/
├── hunter_se/                          # 원본 로봇 USD 에셋 (URDF→USD 변환)
├── hunter_se_v0/                       # ★ RL 학습용 로봇 모델 (공식 메뉴얼 스펙)
│   ├── hunter_se_v0_cfg.py             # ArticulationCfg (검증된 물리 파라미터)
│   ├── hunter_se_v0.usda               # 기본 도형(Box/Sphere) 기반 procedural USD
│   ├── build_usd.py                    # USDA 자동 생성 스크립트
│   └── ackermann.py                    # Ackermann 조향 계산 (메뉴얼 스펙 적용)
├── source/
│   ├── isaaclab_autodrive/             # 공통 모듈 패키지 (로봇 에셋, 유틸)
│   │   └── assets/robots/hunter.py    # HUNTER_SE_CFG / HUNTER_SE_V0_CFG
│   └── isaaclab_autodrive_tasks/       # RL 환경 태스크 패키지
│       └── direct/
│           ├── lidar_nav/              # ★ 주력 태스크 (LiDAR 자율주행, hunter_se_v0 기반)
│           └── legacy/                 # 보존용 path tracking 계열
├── scripts/
│   └── autodrive/
│       ├── train_tqc.py                # 학습 메인 스크립트 (TQC / TD7)
│       ├── play_lidar_nav.py           # 정책 평가/시각화
│       ├── benchmark.py                # 알고리즘 비교
│       ├── teleop_hunter_se_v0.py      # Hunter SE V0 키보드 텔레오퍼레이션
│       ├── drive_straight_hunter_se_v0.py  # 최고속도 직진 검증 (1.333 m/s)
│       ├── teleop_mushr_sus.py         # MuSHR Nano SUS 키보드 텔레오퍼레이션
│       ├── drive_hunter_se.py          # Hunter SE (원본) Ackermann 수동 주행 검증
│       ├── spawn_hunter_se.py          # 로봇 스폰 검증
│       └── algorithms/                 # TQC / TD7 / SAC 구현체
│           ├── tqc/
│           ├── td7/
│           ├── sac/
│           └── common/
├── apps/                               # Isaac Sim 실행 kit 설정
├── logs/                               # 학습 로그 출력 디렉터리
└── ros2/                               # [Phase 4 예정] 실로봇 ROS2 배포
```

---

## 빠른 시작

### 0. Setup
```bash
# Host
sudo systemctl restart docker
xhost +local:
```

### 1. 패키지 설치

```bash
cd /workspace/isaaclab
./isaaclab.sh -i

cd /workspace/hunter_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```

### 2. 학습 실행

```bash
# Phase C TQC 학습 (해석적 LiDAR, 64 envs)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc --num_envs 64 --headless

# Phase E TQC 학습 (물리 벽 + 물리 장애물)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-PhaseE-v0 \
    --algo tqc --num_envs 64 --headless

# TD7 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo td7 --num_envs 64 --headless
```

### 3. 정책 평가

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-Play-v0 \
    --algo tqc \
    --checkpoint logs/tqc/hunter_tqc/TIMESTAMP/model_final.pt \
    --num_envs 4
```

### 4. 알고리즘 비교

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --tqc_ckpt logs/tqc/.../model_final.pt \
    --td7_ckpt logs/td7/.../model_final.pt
```

### 5. 로봇 텔레오퍼레이션 / 검증

```bash
# Hunter SE V0 키보드 조종
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_hunter_se_v0.py

# 최고속도 직진 물리 검증 (헤드리스 가능)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_straight_hunter_se_v0.py --headless

# MuSHR Nano SUS 키보드 조종
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_mushr_sus.py
```

---

## train_tqc.py 인수

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--algo` | `tqc` | 알고리즘 (`tqc` / `td7`) |
| `--task` | `Isaac-LidarNav-Hunter-v0` | Gymnasium 환경 ID |
| `--num_envs` | cfg 기본값 | 병렬 환경 수 |
| `--cfg` | 태스크 내 기본 YAML | 커스텀 하이퍼파라미터 YAML 경로 |
| `--seed` | `0` | 랜덤 시드 |
| `--device` | `cuda:0` | 연산 디바이스 |
| `--headless` | `False` | 헤드리스 실행 여부 |

로그 저장 경로: `logs/{algo}/{experiment_name}/{timestamp}/`

---

## 등록 환경 (Gymnasium)

| 환경 ID | Phase | num_envs | 용도 |
|---|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | C | 64 | 학습 (해석적 LiDAR) |
| `Isaac-LidarNav-Hunter-Play-v0` | C | 4 | 시각화/평가 |
| `Isaac-LidarNav-Hunter-PhaseE-v0` | E | 64 | 학습 (물리 벽 + 물리 장애물) |
| `Isaac-LidarNav-Hunter-PhaseE-Play-v0` | E | 4 | Phase E 시각화/평가 |

모든 환경은 **Hunter SE V0** 로봇 모델(`hunter_se_v0_cfg.py`)을 사용합니다.

---

## 관측 / 행동 공간

| 항목 | 차원 | 내용 |
|---|---|---|
| 관측 (obs) | 82D | LiDAR 80 sector (거리 정규화, [0,1]) + `goal_dist` + `goal_angle` |
| 행동 (act) | 2D | `linear_vel [-1,1]`, `angular_vel [-1,1]` |
| LiDAR | 5.0 m / 80 sector | 360° / 4.5° 간격, 해석적 ray-cylinder + ray-AABB |
| 맵 크기 | 16 × 16 m | Phase E: 물리 벽(4면) + 원통 장애물 5개 |

---

## 보상 함수

```
r = goal_progress(Δdist) × 5.0       # dense, 목표 접근
  + goal_reached × 100.0              # sparse, 목표 도달
  + collision × (−10.0)               # sparse, 충돌
  − proximity_penalty(min_dist)        # zone-based smooth (최대 2.0)
  − 0.01                              # time penalty per step
```

종료 조건:
- `goal_dist < 0.3 m` → 목표 도달 (success)
- 로봇-장애물 거리 < `ROBOT_RADIUS + obs_radius` → 충돌 (failure)
- 맵 이탈 `(|x| > 8m or |y| > 8m)` → 범위 초과 (Phase C only)
- `episode_steps >= max_steps` → 타임아웃

---

## 로봇 모델 (Hunter SE V0)

RL 학습 환경에서 사용하는 `hunter_se_v0` 모델의 주요 제원:

| 항목 | 값 | 비고 |
|---|---|---|
| 자체 중량 | 42 kg | 공식 메뉴얼 기준 |
| 최고 속도 | 1.333 m/s (4.8 km/h) | 공식 메뉴얼 기준 |
| 축거 (wheelbase) | 0.548 m | URDF 관절 위치 기준 |
| 윤거 (후륜) | 0.504 m | |
| 바퀴 반지름 | 0.1375 m | 공식 메뉴얼 직경 0.275 m / 2 |
| 최대 조향각 | ±0.384 rad (±22°) | Ackermann 중심각 기준 |

액추에이터 파라미터 (검증값):

| 파라미터 | 값 | 근거 |
|---|---|---|
| 후륜 `effort_limit` | 11.0 N·m | 슬립 한계(13.94 N·m) 대비 21% 여유 |
| 후륜 `velocity_limit` | 15.0 rad/s | 크루즈 속도에서 포화 토크 여유 확보 |
| 후륜 `damping` | 15.0 N·m·s/rad | 50 Hz 제어 루프 이산 극점 +0.295 (진동 없음) |
| 전륜 `damping` | 0.01 N·m·s/rad | 실제 베어링 마찰 수준 (전진 저항 1.4 N) |

---

## 지원 알고리즘

| 알고리즘 | 유형 | 특징 |
|---|---|---|
| TQC | Off-policy | Quantile 분산 RL, 상위 분위수 제거로 과대추정 억제 — **주력** |
| TD7 | Off-policy | SALE 인코더, LAP PER, 성능 회귀 복원 체크포인팅 |
| SAC | Off-policy | 엔트로피 정규화 기반 탐색 |

---

## 개발 단계

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 로봇 USD 모델 검증, Ackermann 수동 주행, TQC/TD7/SAC 이식 | ✅ 완료 |
| 2 | Path tracking 환경 (legacy) | ✅ 완료 |
| 3-A | Legacy 코드 격리 | ✅ 완료 |
| 3-B | lidar_nav 환경 골격 + Gymnasium 등록 | ✅ 완료 |
| 3-C | 해석적 LiDAR + 실제 관측/보상/종료 | ✅ 완료 |
| 3-D | TQC/TD7 학습 루프 연결, play/benchmark 스크립트 | ✅ 완료 |
| 3-E | 물리 벽 + 물리 장애물, ray-AABB LiDAR 확장 | ✅ 완료 |
| 3-F | Hunter SE V0 물리 파라미터 검증 및 RL 환경 적용 | ✅ 완료 |
| 4 | ONNX 내보내기 + ROS2 브리지, 실로봇 배포 | 🔜 예정 |

---

## 의존성

| 항목 | 경로 / 버전 |
|---|---|
| Isaac Lab | `/workspace/isaaclab/` — **v2.3.2** |
| Isaac Sim | `/isaac-sim/` — **v5.0.0-rc.45** |
| 본 프로젝트 | `/workspace/hunter_autodrive/` |
| Hunter SE V0 모델 | `/workspace/hunter_autodrive/hunter_se_v0/` |
| Hunter SE 원본 USD | `/workspace/hunter_autodrive/hunter_se/` |
| ROS2 워크스페이스 (예정) | `/robot_isaac/ros2_ws/` |
