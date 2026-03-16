# Hunter SE Autonomous Driving RL

Hunter SE 로봇을 대상으로 Isaac Lab / Isaac Sim 기반 강화학습으로 자율주행을 학습하고, 학습된 정책을 실로봇에 배포하는 연구 프로젝트입니다.

> **논문 참고:** IEEE/ASME AIM 2024 — *Rough Terrain Path Tracking of an Ackermann Steered Platform using Hybrid Deep Reinforcement Learning*

---

## 환경 구성

| 항목 | 경로 / 버전 |
|---|---|
| Isaac Lab (프레임워크) | `/workspace/isaaclab/` |
| 본 프로젝트 (커스텀 코드) | `/workspace/hunter_autodrive/` |
| Hunter USD 에셋 | `/robot_isaac/ros2_ws/src/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/omniisaacgymenvs/USD_Files/hunter_aim4.usd` |
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
│                                                             │
│  /workspace/isaaclab/                  ← Isaac Lab (의존성)  │
│                                                             │
│  학습 스크립트: scripts/autodrive/                            │
└─────────────────────────────────────────────────────────────┘
                        ↓ ONNX / TorchScript 내보내기
┌─────────────────────────────────────────────────────────────┐
│                   배포 단계 (ROS2 + 실로봇)                   │
│                                                             │
│  /workspace/hunter_autodrive/ros2/                          │
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
├── source/
│   │
│   ├── isaaclab_autodrive/                      # [패키지 1] 핵심 자율주행 모듈
│   │   ├── config/
│   │   │   └── extension.toml                   # 패키지 메타: deps=[isaaclab]
│   │   ├── setup.py
│   │   ├── docs/
│   │   └── isaaclab_autodrive/
│   │       ├── __init__.py
│   │       ├── assets/
│   │       │   ├── __init__.py
│   │       │   └── robots/
│   │       │       ├── __init__.py
│   │       │       └── hunter.py                # HUNTER_CFG (ArticulationCfg)
│   │       │                                    #   USD: hunter_aim4.usd
│   │       │                                    #   휠: velocity 제어 (re_.*)
│   │       │                                    #   조향: position 제어 (fr_.*)
│   │       │                                    #   축거: L=0.608m
│   │       ├── terrains/
│   │       │   ├── __init__.py
│   │       │   ├── track/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── track_terrain_cfg.py     # 평탄 트랙 지형
│   │       │   │   └── waypoints/
│   │       │   │       ├── austin_centerline2.csv
│   │       │   │       ├── brandshatch_centerline.csv
│   │       │   │       └── silverstone_centerline.csv
│   │       │   └── rough/
│   │       │       ├── __init__.py
│   │       │       └── rough_terrain_cfg.py     # 험로 지형 (terraintrain_9_uneven.usd)
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── cubic_spline.py              # CubicSpline1D / CubicSpline2D / calc_spline_course
│   │           ├── lqr_controller.py            # LQR (DARE 풀이), State 클래스
│   │           └── angle.py                     # angle_mod, rot_mat_2d
│   │
│   └── isaaclab_autodrive_tasks/                # [패키지 2] RL 환경 태스크
│       ├── config/
│       │   └── extension.toml                   # deps=[isaaclab, isaaclab_autodrive]
│       ├── setup.py
│       ├── docs/
│       └── isaaclab_autodrive_tasks/
│           ├── __init__.py
│           └── direct/                          # DirectRLEnv 기반
│               ├── __init__.py
│               │
│               ├── path_tracking/               # [태스크 1] 평탄 경로 추종
│               │   ├── __init__.py              #   gym.register("Isaac-PathTracking-Hunter-v0")
│               │   ├── path_tracking_env_cfg.py #   obs=7D, act=2D, envs=4096, ep=200s
│               │   ├── path_tracking_env.py     #   경로 추종 + Ackermann 조향 적용
│               │   └── agents/
│               │       ├── rsl_rl_ppo_cfg.py    #   PPO: lr=1e-3, hidden=[256,256]
│               │       ├── tqc_cfg.yaml         #   TQC: n_critics=5, n_quantiles=25
│               │       └── td7_cfg.yaml         #   TD7 하이퍼파라미터
│               │
│               ├── rough_terrain_tracking/      # [태스크 2] 험로 경로 추종
│               │   ├── __init__.py              #   gym.register("Isaac-RoughTerrainTracking-Hunter-v0")
│               │   ├── rough_env_cfg.py         #   험로 지형 + 도메인 랜덤화
│               │   ├── rough_env.py
│               │   └── agents/
│               │       ├── rsl_rl_ppo_cfg.py
│               │       └── tqc_cfg.yaml
│               │
│               ├── hybrid_control/              # [태스크 3] 하이브리드 (DRL + LQR)
│               │   ├── __init__.py              #   gym.register("Isaac-HybridControl-Hunter-v0")
│               │   ├── hybrid_env_cfg.py        #   LQR Q=diag(1,10,100,100), R=1
│               │   ├── hybrid_env.py            #   RL 액션 + LQR 조향 보정
│               │   └── agents/
│               │       ├── rsl_rl_ppo_cfg.py
│               │       └── tqc_cfg.yaml
│               │
│               └── multi_track/                 # [태스크 4] 다중 트랙 일반화
│                   ├── __init__.py              #   gym.register("Isaac-MultiTrack-Hunter-v0")
│                   ├── multi_track_env_cfg.py   #   Austin / BrandsHatch / Silverstone 랜덤 전환
│                   ├── multi_track_env.py
│                   └── agents/
│                       └── rsl_rl_ppo_cfg.py
│
├── scripts/
│   └── autodrive/
│       ├── train.py                             # RSL-RL PPO 학습 (--task, --num_envs, --headless)
│       ├── train_tqc.py                         # TQC off-policy 학습
│       ├── play.py                              # 학습된 정책 시각화
│       ├── benchmark.py                         # PPO vs TQC vs TD7 비교
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── tqc/
│       │   │   ├── __init__.py
│       │   │   ├── tqc_agent.py                 # TQC (Actor + Critic + 엔트로피 자동조정)
│       │   │   ├── tqc_trainer.py               # Isaac Lab 환경 연동 학습 루프
│       │   │   └── networks.py                  # Actor (Gaussian), Critic (Quantile)
│       │   ├── td7/
│       │   │   ├── __init__.py
│       │   │   ├── td7_agent.py
│       │   │   └── td7_trainer.py
│       │   ├── sac/
│       │   │   ├── __init__.py
│       │   │   └── sac_agent.py
│       │   └── common/
│       │       ├── __init__.py
│       │       ├── buffer.py                    # LAP Prioritized Experience Replay
│       │       └── logger.py                    # TensorBoard + JSON 로거
│       └── deploy/
│           ├── export_policy.py                 # ONNX / TorchScript 내보내기
│           ├── test_policy.py                   # 로컬 정책 테스트
│           └── ros2_node_template.py            # ROS2 배포 노드 템플릿
│
└── ros2/
    ├── autodrive_interfaces/                    # [ROS2 패키지 1] 커스텀 인터페이스
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── srv/
    │   │   ├── GetAction.srv                    # float32[] state → float32[] action
    │   │   ├── LoadPolicy.srv                   # string model_path → bool success
    │   │   ├── SetTrack.srv                     # string track_name → bool success
    │   │   └── GetStatus.srv                    # () → string mode, int32 episode, int32 steps
    │   └── action/
    │       └── RunEpisode.action                # goal: mode / feedback: step,reward / result: total_reward
    │
    └── isaaclab_ros2_bridge/                    # [ROS2 패키지 2] 실로봇 배포 브리지
        ├── CMakeLists.txt
        ├── package.xml
        ├── launch/
        │   ├── deploy_hunter.launch.py          # 실로봇 배포 런치
        │   └── sim_validate.launch.py           # 시뮬-실환경 검증
        ├── config/
        │   ├── policy.yaml                      # 정책 파일 경로 및 입출력 설정
        │   └── robot.yaml                       # 로봇 토픽 매핑 (/odom, /scan, /cmd_vel)
        └── isaaclab_ros2_bridge/
            ├── __init__.py
            ├── policy_node.py                   # ONNX 정책 로드 → /cmd_vel 발행
            ├── state_processor.py               # /odom + /scan → 7D 상태 변환
            └── action_converter.py              # 2D 액션 → Hunter Ackermann 조향/속도 변환
```

---

## 관측/행동 공간

| 항목 | 차원 | 내용 |
|---|---|---|
| 관측 (obs) | 7D | `x, y, crosstrack_error, heading_error, roll, yaw, linear_velocity` |
| 행동 (act) | 2D | `velocity [-1, 1] m/s`, `steering_angle [-0.524, 0.524] rad` |
| 조향 변환 | Ackermann | `delta_in / delta_out` 좌우 독립 조향각 계산 |

---

## 보상 함수

```
total_reward = exp(-crosstrack_error/5.0) × exp(-heading_error/π) × (0.1 × velocity/3.0)
종료 조건: crosstrack_error ≥ 5.0m  또는  velocity ≤ 0.01 m/s
```

---

## 지원 알고리즘

| 알고리즘 | 유형 | 특징 |
|---|---|---|
| PPO (RSL-RL) | On-policy | 빠른 학습, 4096 병렬 환경 활용 |
| TQC | Off-policy | Quantile 분산 RL, 안정적 수렴 |
| TD7 | Off-policy | LAP 우선순위 버퍼, 고성능 |
| SAC | Off-policy | 엔트로피 정규화, 탐색 효율 우수 |

---

## 트랙 데이터

| 트랙 | 파일 | 특징 |
|---|---|---|
| Austin | `austin_centerline2.csv` | 기본 학습용 |
| Brands Hatch | `brandshatch_centerline.csv` | 코너링 집중 |
| Silverstone | `silverstone_centerline.csv` | 고속 주행 |

---

## 이식 출처

| 이식 대상 | 원본 출처 |
|---|---|
| `assets/robots/hunter.py` | `Hybrid_DRL_Deployments/.../hunter.py` |
| `utils/cubic_spline.py` | `Hybrid_DRL_Deployments/.../CubicSpline.py` |
| `utils/lqr_controller.py` | `Hybrid_DRL_Deployments/.../LQRController.py` |
| `utils/angle.py` | `Hybrid_DRL_Deployments/.../angle.py` |
| `algorithms/tqc/tqc_agent.py` | `drl_agent/scripts/policy/tqc_agent.py` |
| `algorithms/td7/td7_agent.py` | `drl_agent/scripts/policy/td7_agent.py` |
| `algorithms/common/buffer.py` | `drl_agent/scripts/utils/buffer.py` |
| `ros2/autodrive_interfaces/` | `drl_agent_interfaces/` (확장) |
| `ros2/.../state_processor.py` | `drl_agent/scripts/environment/environment.py` (참고) |

---

## 개발 단계

### Phase 1 — 기반 구축
- [x] `isaaclab_autodrive` 패키지 생성 및 `pip install -e` 등록
- [x] `hunter.py` 이식 (USD 경로: `hunter_aim4.usd`)
- [x] `cubic_spline.py`, `lqr_controller.py`, `angle.py` 이식
- [x] 트랙 CSV 복사 (`waypoints/`) — Austin 4145개 경로점 확인
- [x] 기본 임포트 테스트 통과

### Phase 2 — 기본 태스크 구현
- [x] `path_tracking_env.py` 구현 (`hunter_hybrid_env.py` 리팩토링)
- [x] RSL-RL PPO로 Austin 트랙 학습 확인
- [x] `rough_terrain_tracking` 태스크 추가
- [x] TensorBoard 로그 및 체크포인트 확인

### Phase 3 — 알고리즘 확장 ✅ 완료 (2026-03-16)
- [x] TQC 알고리즘 Isaac Lab 환경 연동
  - `tqc_agent.py`: Actor (Gaussian) + Quantile Critic, 자동 엔트로피 조정, top-quantile dropping
  - `tqc_trainer.py`: 병렬 환경(num_envs) 배치 처리, 웜업·평가·체크포인트 루프
  - `networks.py`: 2-layer MLP Actor, QuantileCritic (n_critics × n_quantiles)
  - `path_tracking` 태스크: `tqc_cfg.yaml` (n_critics=5, n_quantiles=25, 1M steps)
- [x] TD7 알고리즘 연동
  - `td7_agent.py`: SALE 인코더, 체크포인팅, 성능 회귀 복원(>20% 하락 시 자동 롤백)
  - `td7_trainer.py`: Isaac Lab 환경 연동, LAP max_priority 주기적 리셋
  - `path_tracking` 태스크: `td7_cfg.yaml` (LAP PER 기본 활성화, 40K 체크포인트)
- [x] LAP PER 버퍼 연동
  - `common/buffer.py`: Latent Action Priority PER, `prioritized` 플래그로 ON/OFF
  - TQC(선택적) · TD7(기본 활성) 모두 연동 완료
- [x] PPO vs TQC vs TD7 성능 비교
  - `benchmark.py`: 세 알고리즘 체크포인트 일괄 평가, mean/std/min/max 리포트 + JSON 저장
- [x] `common/logger.py`: TensorBoard + JSON 이중 로깅

> **미비 사항:** `rough_terrain_tracking` 태스크에는 PPO 설정만 존재 (TQC/TD7 YAML 미생성)

### Phase 4 — 하이브리드 제어 및 일반화
- [ ] `hybrid_control` 태스크: LQR + RL 결합 (디렉터리 스켈레톤만 존재)
- [ ] `multi_track` 태스크: 3개 트랙 랜덤 전환 (디렉터리 스켈레톤만 존재)
- [ ] `rough_terrain_tracking` TQC/TD7 에이전트 설정 추가
- [ ] 도메인 랜덤화 추가 (마찰, 센서 노이즈)

### Phase 5 — 실로봇 배포
- [ ] ONNX 정책 내보내기 (`export_policy.py`)
- [ ] `autodrive_interfaces` ROS2 패키지 생성
- [ ] `isaaclab_ros2_bridge` 노드 구현
- [ ] 실로봇 테스트 및 Sim-to-Real 갭 분석

---

## 패키지 설치

```bash
# Isaac Lab 패키지 설치 (의존성)
cd /workspace/isaaclab
./isaaclab.sh -i

# 커스텀 패키지 설치
cd /workspace/hunter_autodrive
./isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
./isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```

## 학습 실행

> 모든 명령은 `/workspace/hunter_autodrive/` 디렉터리에서 실행합니다.

### PPO 학습 (`train.py`)

```bash
# 기본 학습 (4096 환경, 헤드리스)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --num_envs 4096 --headless

# 트랙 선택 (austin / brandshatch / silverstone)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --num_envs 4096 --track brandshatch --headless

# 소규모 디버그
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --num_envs 64 --headless

# 비디오 녹화 포함 학습
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --num_envs 64 --video --video_length 200 --video_interval 2000
```

주요 인수:

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--task` | `Isaac-PathTracking-Hunter-v0` | 학습 태스크 |
| `--num_envs` | cfg 기본값 | 병렬 환경 수 |
| `--max_iterations` | cfg 기본값 | 학습 반복 횟수 |
| `--track` | `austin` | 트랙 선택 (`austin` / `brandshatch` / `silverstone`) |
| `--seed` | cfg 기본값 | 랜덤 시드 |
| `--headless` | `False` | 헤드리스 실행 |
| `--video` | `False` | 학습 중 비디오 녹화 |

로그 저장 경로: `logs/rsl_rl/{experiment_name}/{timestamp}/`

---

### TQC / TD7 학습 (`train_tqc.py`)

```bash
# TQC 학습 (기본값)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --algo tqc --num_envs 64 --headless

# TD7 학습 (LAP PER 기본 활성화)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --algo td7 --num_envs 64 --headless

# 트랙 및 커스텀 config 지정
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --algo tqc --track silverstone \
    --cfg path/to/tqc_cfg.yaml --headless
```

주요 인수:

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--algo` | `tqc` | 알고리즘 선택 (`tqc` / `td7`) |
| `--task` | `Isaac-PathTracking-Hunter-v0` | 학습 태스크 |
| `--num_envs` | cfg 기본값 | 병렬 환경 수 |
| `--track` | `austin` | 트랙 선택 |
| `--cfg` | 태스크 내 기본 YAML | 커스텀 config 파일 경로 |
| `--seed` | cfg 기본값 | 랜덤 시드 |

로그 저장 경로: `logs/{algo}/{experiment_name}/{timestamp}/`

---

### 시각화 (`play.py`)

```bash
# PPO 정책 시각화 (체크포인트 자동 탐색)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play.py \
    --task Isaac-PathTracking-Hunter-Play-v0 \
    --num_envs 16

# 체크포인트 직접 지정 + 트랙 선택
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play.py \
    --task Isaac-PathTracking-Hunter-Play-v0 \
    --num_envs 16 --track brandshatch \
    --checkpoint logs/rsl_rl/hunter_path_tracking/2026-.../model_1000.pt

# 비디오 저장
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play.py \
    --task Isaac-PathTracking-Hunter-Play-v0 \
    --num_envs 16 --video --video_length 200
```

> **주의:** play 태스크는 `Isaac-PathTracking-Hunter-Play-v0` 사용 (train 태스크와 별도)

---

### 알고리즘 비교 벤치마크 (`benchmark.py`)

```bash
# PPO / TQC / TD7 전체 비교
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --ppo_ckpt logs/rsl_rl/hunter_path_tracking/2026-.../model_0.pt \
    --tqc_ckpt logs/tqc/hunter_path_tracking_tqc/.../model_final.pt \
    --td7_ckpt logs/td7/hunter_path_tracking_td7/.../model_final.pt \
    --num_envs 16 --eval_episodes 20

# 특정 알고리즘만 평가
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --tqc_ckpt logs/tqc/.../model_final.pt \
    --num_envs 16

# 결과 JSON 저장 경로 지정
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --ppo_ckpt ... --tqc_ckpt ... --td7_ckpt ... \
    --output outputs/my_benchmark.json
```

주요 인수:

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--ppo_ckpt` | `None` | PPO 체크포인트 경로 (`.pt`) |
| `--tqc_ckpt` | `None` | TQC 체크포인트 경로 (`.pt`) |
| `--td7_ckpt` | `None` | TD7 체크포인트 경로 (`.pt`) |
| `--eval_episodes` | `20` | 알고리즘당 평가 에피소드 수 |
| `--max_steps` | `2000` | 에피소드당 최대 스텝 |
| `--track` | `austin` | 평가 트랙 |
| `--output` | `outputs/benchmark_{ts}.json` | 결과 JSON 저장 경로 |

결과는 콘솔 테이블(mean/std/min/max) 및 JSON으로 자동 저장됩니다.

## ROS2 배포

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
