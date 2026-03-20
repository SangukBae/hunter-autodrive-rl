# Hunter SE Autonomous Driving RL

Hunter SE 모바일 로봇을 대상으로 **Isaac Lab 기반 강화학습**으로 LiDAR 기반 자율주행(목표 도달 + 장애물 회피)을 학습하고 실로봇에 배포하는 프로젝트입니다.

---

## 폴더 구조

```
hunter_autodrive/
├── hunter_se/                          # 로봇 USD 에셋 + ArticulationCfg
├── source/
│   ├── isaaclab_autodrive/             # 공통 모듈 패키지 (로봇 에셋, 유틸)
│   └── isaaclab_autodrive_tasks/       # RL 환경 태스크 패키지
│       └── direct/
│           ├── lidar_nav/              # ★ 주력 태스크 (LiDAR 자율주행)
│           └── legacy/                 # 보존용 path tracking 계열
├── scripts/
│   └── autodrive/
│       ├── train_tqc.py                # 학습 메인 스크립트
│       ├── play_lidar_nav.py           # 정책 평가/시각화
│       ├── benchmark.py                # 알고리즘 비교
│       ├── spawn_hunter_se.py          # 로봇 스폰 검증
│       ├── drive_hunter_se.py          # Ackermann 수동 주행 검증
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
| `Isaac-LidarNav-Hunter-PhaseE-Play-v0` | E | 4 | PhaseE 시각화/평가 |

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

## 지원 알고리즘

| 알고리즘 | 유형 | 특징 |
|---|---|---|
| TQC | Off-policy | Quantile 분산 RL, 상위 분위수 제거로 과대추정 억제 — **주력** |
| TD7 | Off-policy | SALE 인코더, LAP PER, 성능 회귀 복원 체크포인팅 |
| SAC | Off-policy | 엔트로피 정규화 기반 탐색 |
| PPO (RSL-RL) | On-policy | legacy |

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
| 4 | ONNX 내보내기 + ROS2 브리지, 실로봇 배포 | 🔜 예정 |

---

## 의존성

| 항목 | 경로 / 버전 |
|---|---|
| Isaac Lab | `/workspace/isaaclab/` |
| 본 프로젝트 | `/workspace/hunter_autodrive/` |
| Hunter SE USD | `/workspace/hunter_autodrive/hunter_se/` |
| ROS2 워크스페이스 (예정) | `/robot_isaac/ros2_ws/` |
