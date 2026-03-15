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
- [ ] `path_tracking_env.py` 구현 (`hunter_hybrid_env.py` 리팩토링)
- [ ] RSL-RL PPO로 Austin 트랙 학습 확인
- [ ] `rough_terrain_tracking` 태스크 추가
- [ ] TensorBoard 로그 및 체크포인트 확인

### Phase 3 — 알고리즘 확장
- [ ] TQC 알고리즘 Isaac Lab 환경 연동
- [ ] TD7 알고리즘 연동
- [ ] LAP PER 버퍼 연동
- [ ] PPO vs TQC vs TD7 성능 비교

### Phase 4 — 하이브리드 제어 및 일반화
- [ ] `hybrid_control` 태스크: LQR + RL 결합
- [ ] `multi_track` 태스크: 3개 트랙 랜덤 전환
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

```bash
# PPO 학습
./isaaclab.sh -p scripts/autodrive/train.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --num_envs 4096 --headless

# TQC 학습
./isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-PathTracking-Hunter-v0

# 시각화
./isaaclab.sh -p scripts/autodrive/play.py \
    --task Isaac-PathTracking-Hunter-v0 \
    --num_envs 16
```

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
