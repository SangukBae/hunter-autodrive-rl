# Hunter SE Autonomous Driving RL

Hunter SE V0 모바일 로봇을 대상으로 **Isaac Lab 기반 강화학습**으로 LiDAR 기반 자율주행(목표 도달 + 장애물 회피)을 학습하는 프로젝트입니다.

- **로봇**: Hunter SE V0 (`hunter_se_v0/`) — 공식 메뉴얼 스펙 기반, 물리 파라미터 검증 완료
- **LiDAR**: RTX OmniLidar (OS1-32) — 실물과 동일한 물리 기반 광자 추적 방식
- **맵**: 16×16m 물리 벽 경계, 매 에피소드 10개 장애물 랜덤 배치
- **충돌 판정**: LiDAR 최소 감지 거리 기반

---

## 폴더 구조

```
hunter_autodrive/
├── hunter_se_v0/                       # RL 학습용 로봇 모델 (공식 메뉴얼 스펙)
│   ├── hunter_se_v0_cfg.py             # ArticulationCfg (검증된 물리 파라미터)
│   ├── hunter_se_v0.usda               # procedural USD (기본 도형 기반)
│   ├── build_usd.py                    # USDA 자동 생성 스크립트
│   ├── ackermann.py                    # Ackermann 조향 계산
│   └── lidar_cfg.py                    # RTX LiDAR 모델 사양 정의
├── source/
│   ├── isaaclab_autodrive/             # 공통 모듈 패키지 (로봇 에셋)
│   │   └── assets/robots/hunter.py    # HUNTER_SE_V0_CFG 재익스포트
│   └── isaaclab_autodrive_tasks/       # RL 환경 태스크 패키지
│       └── direct/lidar_nav/           # LiDAR 자율주행 태스크
├── scripts/
│   └── autodrive/
│       ├── train_tqc.py                # 학습 메인 스크립트 (TQC / TD7)
│       ├── play_lidar_nav.py           # 정책 평가/시각화
│       ├── test_lidar_nav.py           # 환경 단위 테스트
│       ├── teleop_lidar_check.py       # LiDAR 동작 확인
│       └── algorithms/                 # TQC / TD7 구현체
│           ├── tqc/
│           ├── td7/
│           └── common/
├── apps/                               # Isaac Sim 실행 kit 설정
└── logs/                               # 학습 로그 출력 디렉터리
```

---

## 빠른 시작

### 1. 패키지 설치 (최초 1회)

```bash
cd /workspace/hunter_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```

### 2. 학습 실행

```bash
# TQC 학습 (RTX LiDAR, 16×16 벽+장애물 맵)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc --num_envs 4 --headless

# TD7 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo td7 --num_envs 4 --headless
```

> RTX LiDAR는 환경당 독립 render product를 생성하므로 num_envs 4~8 권장.

### 3. 정책 평가

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-Play-v0 \
    --algo tqc \
    --checkpoint logs/tqc/hunter_tqc/TIMESTAMP/model_final.pt \
    --num_envs 4
```

### 4. 환경 테스트 / LiDAR 확인

```bash
# 환경 단위 테스트
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/test_lidar_nav.py

# LiDAR 동작 확인
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/teleop_lidar_check.py
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

| 환경 ID | num_envs | 용도 |
|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | 64 | 학습 (RTX LiDAR, 벽+장애물) |
| `Isaac-LidarNav-Hunter-Play-v0` | 4 | 시각화/평가 |

---

## 관측 / 행동 공간

| 항목 | 차원 | 내용 |
|---|---|---|
| 관측 (obs) | 82D | LiDAR 80 sector (거리 정규화, [0,1]) + `goal_dist` + `goal_angle` |
| 행동 (act) | 2D | `linear_vel [-1,1]`, `angular_vel [-1,1]` |
| LiDAR | OS1-32, 120m / 80 sector | RTX OmniLidar, 360° / 4.5° 간격 |
| 맵 크기 | 16×16m | 물리 벽(4면) + 원통 장애물 10개 (랜덤) |

---

## 보상 함수

```
r = goal_progress(Δdist) × 5.0       # dense, 목표 접근
  + goal_reached × 100.0              # sparse, 목표 도달
  + collision × (−10.0)               # sparse, 충돌
  − proximity_penalty(min_lidar_dist)  # smooth (최대 2.0)
  − 0.01                              # time penalty per step
```

종료 조건:
- `goal_dist < 0.3 m` → 목표 도달 (success)
- LiDAR 최소 감지 거리 < 0.3 m → 충돌 (failure)
- `episode_steps >= max_steps` → 타임아웃

---

## 로봇 모델 (Hunter SE V0)

| 항목 | 값 |
|---|---|
| 자체 중량 | 42 kg |
| 최고 속도 | 1.333 m/s (4.8 km/h) |
| 축거 (wheelbase) | 0.548 m |
| 윤거 (후륜) | 0.504 m |
| 바퀴 반지름 | 0.1375 m |
| 최대 조향각 | ±0.384 rad (±22°) |
| 스폰 높이 | z = 0.2955 m |

---

## 지원 알고리즘

| 알고리즘 | 유형 | 특징 |
|---|---|---|
| TQC | Off-policy | Quantile 분산 RL, 상위 분위수 제거로 과대추정 억제 — **주력** |
| TD7 | Off-policy | SALE 인코더, LAP PER, 성능 회귀 복원 체크포인팅 |

---

## 의존성

| 항목 | 경로 / 버전 |
|---|---|
| Isaac Lab | `/workspace/isaaclab/` — v2.3.2 |
| Isaac Sim | `/isaac-sim/` — v5.0.0-rc.45 |
| 본 프로젝트 | `/workspace/hunter_autodrive/` |
| Hunter SE V0 모델 | `/workspace/hunter_autodrive/hunter_se_v0/` |
