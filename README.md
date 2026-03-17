# Hunter SE Autonomous Driving RL

Hunter SE 로봇을 대상으로 Isaac Lab / Isaac Sim 기반 강화학습으로 **LiDAR 기반 자율주행(goal-reaching + obstacle avoidance)** 을 학습하고, 학습된 정책을 실로봇에 배포하는 연구 프로젝트입니다.

> **연구 주제:** LiDAR 기반 실내 모바일 로봇 자율주행 — 목표 지점 도달 + 장애물 회피
> **핵심 알고리즘:** TQC (Truncated Quantile Critics), TD7
> **시뮬레이터:** Isaac Lab (Isaac Sim 4.x 기반)

---

## 환경 구성

| 항목 | 경로 / 버전 |
|---|---|
| Isaac Lab (프레임워크) | `/workspace/isaaclab/` |
| 본 프로젝트 (커스텀 코드) | `/workspace/hunter_autodrive/` |
| Hunter SE USD 에셋 | `/workspace/hunter_autodrive/hunter_se/` |
| ROS2 워크스페이스 | `/robot_isaac/ros2_ws/` |

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    학습 단계 (Isaac Lab)                      │
│                                                             │
│  /workspace/hunter_autodrive/                               │
│  ├── source/isaaclab_autodrive/        ← 핵심 모듈           │
│  └── source/isaaclab_autodrive_tasks/  ← RL 환경 태스크       │
│       └── direct/lidar_nav/           ← [주력] LiDAR 자율주행 │
│                                                             │
│  /workspace/isaaclab/                  ← Isaac Lab (의존성)  │
│                                                             │
│  학습 스크립트: scripts/autodrive/                            │
└─────────────────────────────────────────────────────────────┘
                        ↓ ONNX / TorchScript 내보내기
┌─────────────────────────────────────────────────────────────┐
│                   배포 단계 (ROS2 + 실로봇)                   │
│                                                             │
│  /workspace/hunter_autodrive/ros2/          ← 계획됨 (Phase 4) │
│  ├── autodrive_interfaces/   ← 커스텀 ROS2 인터페이스         │
│  └── isaaclab_ros2_bridge/   ← 학습된 정책 실행 노드          │
│                                                             │
│  (빌드 시 /robot_isaac/ros2_ws/src/ 에 심볼릭 링크)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 폴더 구조

```
hunter_autodrive/
│
├── hunter_se/                               # [로봇 자산] Hunter SE 물리 모델
│   ├── hunter_se_description.usda           #   최상위 USD 기술자
│   ├── hunter_se_cfg.py                     #   ArticulationCfg (Isaac Lab)
│   └── Payload/
│       ├── Physics.usda                     #   관절 물리 설정 (검증 완료)
│       ├── Geometry.usda                    #   충돌/시각 메시
│       └── Materials.usda
│
├── source/
│   │
│   ├── isaaclab_autodrive/                  # [패키지 1] 핵심 자율주행 모듈
│   │   └── isaaclab_autodrive/
│   │       ├── assets/robots/hunter.py      #   HUNTER_SE_CFG (hunter_se/ 참조)
│   │       ├── terrains/                    #   [legacy] track/rough 지형
│   │       └── utils/
│   │           ├── angle.py                 #   angle_mod, rot_mat_2d (재사용)
│   │           ├── cubic_spline.py          #   [legacy] 경로 추종용
│   │           └── lqr_controller.py        #   [legacy] LQR 제어기
│   │
│   └── isaaclab_autodrive_tasks/            # [패키지 2] RL 환경 태스크
│       └── isaaclab_autodrive_tasks/
│           └── direct/
│               │
│               ├── lidar_nav/              # ★ [주력 태스크] LiDAR 자율주행
│               │   ├── __init__.py         #   4개 env 등록
│               │   │                       #     Isaac-LidarNav-Hunter-v0
│               │   │                       #     Isaac-LidarNav-Hunter-Play-v0
│               │   │                       #     Isaac-LidarNav-Hunter-PhaseE-v0
│               │   │                       #     Isaac-LidarNav-Hunter-PhaseE-Play-v0
│               │   ├── lidar_nav_env_cfg.py#   cfg 클래스 4종 (Phase C / Phase E)
│               │   ├── lidar_nav_env.py    #   환경 본체 (Phase C + E 통합)
│               │   └── agents/
│               │       ├── tqc_cfg.yaml    #   TQC: n_critics=5, n_quantiles=25
│               │       └── td7_cfg.yaml    #   TD7: LAP PER 기본 활성
│               │
│               └── legacy/                 # [legacy] path tracking 계열 (보존)
│                   ├── path_tracking/      #   crosstrack error 기반 경로 추종
│                   ├── rough_terrain_tracking/
│                   ├── hybrid_control/
│                   └── multi_track/
│
├── scripts/
│   └── autodrive/
│       ├── train_tqc.py                    # TQC / TD7 off-policy 학습 메인
│       ├── train.py                        # RSL-RL PPO 학습 (legacy)
│       ├── play_lidar_nav.py               # TQC/TD7 정책 시각화 및 평가
│       ├── play.py                         # RSL-RL 정책 시각화 (legacy)
│       ├── benchmark.py                    # 알고리즘 비교 평가
│       ├── spawn_hunter_se.py              # 로봇 단독 스폰/검증
│       ├── drive_hunter_se.py              # Ackermann 수동 주행 검증
│       ├── algorithms/
│       │   ├── tqc/
│       │   │   ├── networks.py             #   Actor (Gaussian) + QuantileCritic
│       │   │   ├── tqc_agent.py            #   TQC 에이전트
│       │   │   └── tqc_trainer.py          #   Isaac Lab 환경 연동 학습 루프
│       │   ├── td7/
│       │   │   ├── td7_agent.py            #   SALE 인코더, LAP PER, Checkpointing
│       │   │   └── td7_trainer.py
│       │   ├── sac/
│       │   │   └── sac_agent.py
│       │   └── common/
│       │       ├── buffer.py               #   LAP Prioritized Experience Replay
│       │       └── logger.py              #   TensorBoard + JSON 로거
│       └── deploy/
│           └── export_policy.py            #   ONNX / TorchScript 내보내기 (계획됨)
│
├── apps/
│   ├── isaaclab.python.kit                 # Isaac Lab GUI 실행 구성
│   └── isaaclab.python.headless.kit        # Isaac Lab 헤드리스 실행 구성
│
└── ros2/                                   # [계획됨 — Phase 4]
    ├── autodrive_interfaces/               #   커스텀 ROS2 인터페이스 (srv, action)
    └── isaaclab_ros2_bridge/               #   실로봇 배포 브리지 노드
```

---

## 관측 / 행동 공간

| 항목 | 차원 | 내용 |
|---|---|---|
| 관측 (obs) | 82D | LiDAR 80 sector (거리 정규화) + `[goal_dist, goal_angle]` |
| 행동 (act) | 2D | `linear_vel [-1, 1]`, `angular_vel [-1, 1]` |
| LiDAR 범위 | 5.0 m | 360° / 80 sector (4.5° 간격) |
| 맵 크기 | 16 × 16 m | Phase E: 물리 벽 + 원통 장애물 N개 |

---

## 보상 함수

```
r_step = goal_progress(Δdist) × k_p          # 목표 접근 보상 (dense)
       + goal_reached × R_goal                # 목표 도달 보상 (sparse, +100)
       + collision × P_col                    # 충돌 페널티 (sparse, -10)
       + obstacle_proximity(min_dist)         # 근접 페널티 (zone-based, smooth)
       + time_penalty                         # 생존 비용 (-0.01/step)

종료 조건:
  - goal_dist < 0.3 m               → goal reached
  - robot ↔ obstacle dist < 0.6 m   → collision
  - map 이탈 (Phase C only)          → out of bounds
  - episode_steps >= max_steps       → timeout
```

---

## 등록 환경 (Gymnasium)

| 환경 ID | Phase | 설명 | num_envs |
|---|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | C | 학습 기본 (해석적 LiDAR) | 64 |
| `Isaac-LidarNav-Hunter-Play-v0` | C | 시각화/평가 | 4 |
| `Isaac-LidarNav-Hunter-PhaseE-v0` | E | 물리 벽 + 물리 장애물 | 64 |
| `Isaac-LidarNav-Hunter-PhaseE-Play-v0` | E | PhaseE 시각화/평가 | 4 |

---

## 지원 알고리즘

| 알고리즘 | 유형 | 특징 |
|---|---|---|
| TQC | Off-policy | Quantile 분산 RL, 과대추정 억제 — **주력** |
| TD7 | Off-policy | SALE 인코더, LAP PER, 성능 회귀 복원 |
| SAC | Off-policy | 엔트로피 정규화 |
| PPO (RSL-RL) | On-policy | 빠른 학습, legacy |

---

## 패키지 설치

```bash
cd /workspace/isaaclab
./isaaclab.sh -i

cd /workspace/hunter_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```

---

## 학습 실행

> 모든 명령은 `/workspace/hunter_autodrive/` 디렉터리에서 실행합니다.

### TQC / TD7 학습 (`train_tqc.py`)

```bash
# Phase C TQC 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc --num_envs 64 --headless

# Phase C TD7 학습 (LAP PER 기본 활성화)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo td7 --num_envs 64 --headless

# Phase E TQC 학습 (물리 벽 + 물리 장애물)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-PhaseE-v0 \
    --algo tqc --num_envs 64 --headless

# 커스텀 config 지정
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc \
    --cfg source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/agents/tqc_cfg.yaml \
    --headless
```

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--algo` | `tqc` | 알고리즘 선택 (`tqc` / `td7`) |
| `--task` | `Isaac-LidarNav-Hunter-v0` | 학습 태스크 ID |
| `--num_envs` | cfg 기본값 | 병렬 환경 수 |
| `--cfg` | 태스크 내 기본 YAML | 커스텀 config 파일 경로 |
| `--seed` | 0 | 랜덤 시드 |
| `--headless` | `False` | 헤드리스 실행 |

로그 저장 경로: `logs/{algo}/{experiment_name}/{timestamp}/`

---

### 정책 시각화 및 평가 (`play_lidar_nav.py`)

```bash
# TQC 정책 시각화 (GUI)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-Play-v0 \
    --algo tqc \
    --checkpoint logs/tqc/hunter_tqc/TIMESTAMP/model_final.pt \
    --num_envs 4

# TD7 정책 평가 (헤드리스, 100 에피소드)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-PhaseE-Play-v0 \
    --algo td7 \
    --checkpoint logs/td7/hunter_td7/TIMESTAMP/model_final.pt \
    --eval_episodes 100 --headless
```

---

### 알고리즘 비교 벤치마크 (`benchmark.py`)

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --tqc_ckpt logs/tqc/hunter_tqc/.../model_final.pt \
    --td7_ckpt logs/td7/hunter_td7/.../model_final.pt \
    --num_envs 16 --eval_episodes 20
```

---

## ROS2 배포 (Phase 4 — 계획됨)

```bash
# ros2 패키지를 워크스페이스에 심볼릭 링크
ln -s /workspace/hunter_autodrive/ros2/autodrive_interfaces \
      /robot_isaac/ros2_ws/src/
ln -s /workspace/hunter_autodrive/ros2/isaaclab_ros2_bridge \
      /robot_isaac/ros2_ws/src/

# 빌드
cd /robot_isaac/ros2_ws
colcon build --packages-select autodrive_interfaces isaaclab_ros2_bridge

# 실행
ros2 launch isaaclab_ros2_bridge deploy_hunter.launch.py \
    policy_path:=/path/to/policy.onnx
```

---

## 개발 단계

### Phase 1 — 기반 구축 ✅ 완료
- [x] `isaaclab_autodrive` 패키지 생성 및 `pip install -e` 등록
- [x] `hunter_se/` USD 물리 모델 구축 및 검증 (Physics.usda, hunter_se_cfg.py)
  - `fr_left_joint` 비대칭(localRot1 부호 반전) 수정 → 직진 좌편향 해결
  - 전륜 자유회전 damping 15 → 0.5 (진동 해결)
  - `solver_velocity_iteration_count` 16 → 4 (TGS 안정화)
- [x] `drive_hunter_se.py` 수동 주행 검증 (Ackermann, `STEP_DT` 시간 계산 수정)
- [x] 알고리즘 코어 이식: TQC / TD7 / SAC / LAP PER buffer / TensorBoard logger

### Phase 2 — path tracking 계열 구현 ✅ 완료 (legacy 보존)
- [x] `path_tracking_env.py` (crosstrack error 기반) → `legacy/`로 격리
- [x] `rough_terrain_tracking`, `hybrid_control`, `multi_track` 스켈레톤 → `legacy/`
- [x] RSL-RL PPO / TQC / TD7 연동 완료

### Phase 3 — LiDAR 자율주행 태스크 구성 ✅ 완료

#### Phase A — legacy 격리 ✅
- [x] `direct/path_tracking/` 등 4개 디렉터리 → `direct/legacy/` 하위로 이동
- [x] `direct/__init__.py` 에서 path_tracking 임포트 제거, `lidar_nav` 임포트 추가
- [x] `direct/legacy/__init__.py` 생성 (자동 임포트 없음)

#### Phase B — 환경 골격 생성 ✅
- [x] `lidar_nav/` 디렉터리 + 4개 파일 생성
  (`__init__.py`, `lidar_nav_env_cfg.py`, `lidar_nav_env.py`, `agents/__init__.py`)
- [x] `Isaac-LidarNav-Hunter-v0`, `Isaac-LidarNav-Hunter-Play-v0` gym 등록
- [x] dummy obs (zeros) / dummy reward (0) / timeout-only 종료로 환경 루프 동작 확인
- [x] Ackermann 행동 변환 구현 (Ackermann 내/외륜 분리)

#### Phase C — MVP: 실제 관측/보상/종료 구현 ✅
- [x] **해석적 2D LiDAR**: 80-sector ray-cylinder 교차 계산 (물리 센서 없이 GPU 벡터화)
- [x] **실제 관측 (82D)**: LiDAR 80 sector (정규화) + `goal_dist` + `goal_angle`
- [x] **보상 함수**: `goal_progress` (dense) + `goal_reached` (+100) + `collision` (−10)
  + `obstacle_proximity` (smooth) + `time_penalty` (−0.01/step)
- [x] **종료 조건**: 목표 도달 / 장애물 충돌 / 맵 이탈 / timeout
- [x] **에피소드 리셋**: 로봇 위치·yaw 랜덤화, 목표 위치 랜덤화, 장애물 위치·반경 랜덤화

#### Phase D — TQC / TD7 학습 연결 ✅
- [x] `tqc_cfg.yaml` / `td7_cfg.yaml` — flat key 형식으로 재작성 (Trainer 인터페이스 일치)
- [x] `train_tqc.py` — `--track` 인자 제거, `env_cfg.track_csv` 라인 제거, 기본 태스크를
  `Isaac-LidarNav-Hunter-v0`으로 변경
- [x] `play_lidar_nav.py` 신규 작성 — 결정론적 정책 평가 + 목표 도달률/충돌률 통계 출력

#### Phase E — 논문 환경 재현 (물리 벽 + 물리 장애물) ✅
- [x] **물리 벽 (4개 static cuboid)**: 맵 경계에 스폰 → 로봇의 물리적 map 이탈 방지
- [x] **물리 장애물 (kinematic RigidObject × 5)**: 에피소드마다 위치 재배치
- [x] **LiDAR 벽 감지**: ray-AABB 해석적 교차 추가 → 벽까지 거리 정확히 반영
- [x] `LidarNavEnvCfgPhaseE` 설정 클래스 추가 (`use_walls=True`, `use_physical_obstacles=True`)
- [x] `Isaac-LidarNav-Hunter-PhaseE-v0`, `Isaac-LidarNav-Hunter-PhaseE-Play-v0` 등록

### Phase 4 — 실로봇 배포 🔜 예정
- [ ] ONNX 정책 내보내기 (`export_policy.py` 구현)
- [ ] `isaaclab_ros2_bridge` 노드: `/scan` → 80-sector 상태 변환 → `/cmd_vel`
- [ ] Sim-to-Real 갭 분석 및 도메인 랜덤화

---

## 검증 명령어

> 각 Phase 완료 여부를 확인하는 명령어입니다.
> 모든 명령은 `/workspace/hunter_autodrive/` 에서 실행합니다.

### Phase 1/2 검증 — 로봇 물리 및 기반 구축

```bash
# 로봇 스폰 확인 (관절 구조 + 물리 파라미터 출력)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/spawn_hunter_se.py --headless

# Ackermann 수동 주행 검증
# 기대 동작: 직진 → 좌회전 → 직진 → 우회전 → 정지
# 기대 출력: 초당 속도/위치 로그, 최종 위치 출력
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_hunter_se.py --headless
```

### Phase A 검증 — legacy 격리

```bash
# legacy 디렉터리 구조 확인
ls source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/
# 기대 출력: lidar_nav/  legacy/  __init__.py

ls source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/legacy/
# 기대 출력: path_tracking/  rough_terrain_tracking/  hybrid_control/  multi_track/  __init__.py

# direct/__init__.py 에 lidar_nav만 임포트되는지 확인
grep "import" source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/__init__.py
# 기대 출력: from . import lidar_nav  (path_tracking 없음)
```

### Phase B 검증 — 환경 골격 및 gym 등록

```bash
# gym 등록 환경 목록 확인 (Isaac-LidarNav 4종 출력되어야 함)
/workspace/isaaclab/isaaclab.sh -p - <<'EOF'
import isaaclab_autodrive_tasks
import gymnasium as gym
envs = [k for k in gym.envs.registry.keys() if "LidarNav" in k]
for e in sorted(envs):
    print(e)
EOF
# 기대 출력:
#   Isaac-LidarNav-Hunter-PhaseE-Play-v0
#   Isaac-LidarNav-Hunter-PhaseE-v0
#   Isaac-LidarNav-Hunter-Play-v0
#   Isaac-LidarNav-Hunter-v0
```

### Phase C 검증 — 실제 관측/보상/종료

```bash
# 환경 임포트 + 환경 루프 10 스텝 실행 (헤드리스)
/workspace/isaaclab/isaaclab.sh -p - <<'EOF'
import isaaclab_autodrive_tasks
import gymnasium as gym, torch

env = gym.make("Isaac-LidarNav-Hunter-v0")
obs, _ = env.reset()
print(f"obs shape  : {obs['policy'].shape}")   # 기대: (64, 82)
print(f"obs sample : {obs['policy'][0, :5]}")  # 기대: [0,1] 범위 float

for _ in range(10):
    act = torch.zeros(env.unwrapped.num_envs, 2)
    obs, rew, term, trunc, _ = env.step(act)

print(f"reward sample : {rew[:3]}")            # 기대: 음수(time_penalty 포함)
print(f"obs 82D check : {obs['policy'].shape[-1] == 82}")  # 기대: True
env.close()
EOF
```

### Phase D 검증 — TQC/TD7 학습 연결

```bash
# YAML 키 확인 (flat 형식인지 검증)
python3 -c "
import yaml
with open('source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/agents/tqc_cfg.yaml') as f:
    cfg = yaml.safe_load(f)
required = ['actor_lr', 'n_critics', 'n_quantiles', 'discount', 'total_timesteps']
for k in required:
    assert k in cfg, f'MISSING KEY: {k}'
    print(f'  {k}: {cfg[k]}')
print('YAML OK')
"
# 기대 출력: 각 키 값 출력 후 "YAML OK"

# train_tqc.py --help 확인 (--track 인자 없어야 함)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py --help | grep -E "track|task|algo"
# 기대 출력: --task, --algo 존재 / --track 없음

# play_lidar_nav.py 파일 존재 확인
python3 -c "
import ast, sys
with open('scripts/autodrive/play_lidar_nav.py') as f:
    ast.parse(f.read())
print('play_lidar_nav.py: syntax OK')
"
```

### Phase E 검증 — 물리 벽 + 물리 장애물

```bash
# PhaseE 설정 확인
python3 -c "
import sys; sys.path.insert(0, 'source/isaaclab_autodrive_tasks')
# cfg 임포트 (Isaac Sim 없이 dataclass만 로드)
import importlib
# 플래그 확인
import subprocess
result = subprocess.run(
    ['grep', '-n', 'use_walls\|use_physical_obstacles',
     'source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/lidar_nav_env_cfg.py'],
    capture_output=True, text=True)
print(result.stdout)
"
# 기대 출력: LidarNavEnvCfg에 use_walls=False, LidarNavEnvCfgPhaseE에 use_walls=True

# env.py에 벽 스폰 로직 존재 확인
grep -n "_spawn_walls_per_env\|use_physical_obstacles\|RigidObject\|ray-AABB" \
    source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/lidar_nav_env.py
# 기대 출력: 각 함수/키워드 라인 번호 출력

# PhaseE 환경 gym 등록 확인
python3 -c "
import subprocess
result = subprocess.run(
    ['grep', 'PhaseE',
     'source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/__init__.py'],
    capture_output=True, text=True)
print(result.stdout)
"
# 기대 출력: PhaseE 환경 2종 gym.register 라인 출력
```

### 전체 코드 문법 검증

```bash
# 주요 Python 파일 문법 일괄 검사
python3 -m py_compile \
    source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/lidar_nav_env.py \
    source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/lidar_nav_env_cfg.py \
    source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/__init__.py \
    scripts/autodrive/train_tqc.py \
    scripts/autodrive/play_lidar_nav.py \
    scripts/autodrive/algorithms/tqc/tqc_agent.py \
    scripts/autodrive/algorithms/td7/td7_agent.py \
    scripts/autodrive/algorithms/common/buffer.py \
    && echo "모든 파일 문법 OK"
```

---

## 이식 출처

| 이식 대상 | 원본 출처 |
|---|---|
| `hunter_se/` | URDF → USD 변환 + 물리 파라미터 직접 수정 |
| `assets/robots/hunter.py` | `Hybrid_DRL_Deployments/.../hunter.py` (USD 경로 교체) |
| `algorithms/tqc/tqc_agent.py` | `drl_agent/scripts/policy/tqc_agent.py` |
| `algorithms/td7/td7_agent.py` | `drl_agent/scripts/policy/td7_agent.py` |
| `algorithms/common/buffer.py` | `drl_agent/scripts/utils/buffer.py` (LAP PER) |
| `ros2/autodrive_interfaces/` | `drl_agent_interfaces/` (계획됨) |
| `utils/angle.py` | `Hybrid_DRL_Deployments/.../angle.py` |
