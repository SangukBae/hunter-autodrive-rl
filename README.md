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
│   │   ├── config/
│   │   │   └── extension.toml              #   패키지 메타: deps=[isaaclab]
│   │   ├── setup.py
│   │   └── isaaclab_autodrive/
│   │       ├── assets/
│   │       │   └── robots/
│   │       │       └── hunter.py           #   HUNTER_SE_CFG (hunter_se/ 참조)
│   │       ├── terrains/                   #   [legacy] track/rough 지형
│   │       └── utils/
│   │           ├── angle.py               #   angle_mod, rot_mat_2d (재사용)
│   │           ├── cubic_spline.py        #   [legacy] 경로 추종용
│   │           └── lqr_controller.py      #   [legacy] LQR 제어기
│   │
│   └── isaaclab_autodrive_tasks/            # [패키지 2] RL 환경 태스크
│       ├── config/
│       │   └── extension.toml              #   deps=[isaaclab, isaaclab_autodrive]
│       ├── setup.py
│       └── isaaclab_autodrive_tasks/
│           └── direct/
│               │
│               ├── lidar_nav/              # ★ [주력 태스크] LiDAR 자율주행
│               │   ├── __init__.py         #   gym.register("Isaac-LidarNav-Hunter-v0")
│               │   ├── lidar_nav_env_cfg.py#   obs=82D, act=2D, 맵/보상/LiDAR 파라미터
│               │   ├── lidar_nav_env.py    #   goal-reaching + obstacle avoidance 환경
│               │   ├── observations.py     #   80-sector LiDAR + goal(dist, angle)
│               │   ├── rewards.py          #   goal_progress / goal_reach / collision / proximity
│               │   ├── terminations.py     #   goal / collision / timeout
│               │   ├── randomization.py    #   robot / goal / obstacle 위치 랜덤화
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
│       ├── train.py                        # RSL-RL PPO 학습
│       ├── train_tqc.py                    # TQC / TD7 off-policy 학습
│       ├── play.py                         # 학습된 정책 시각화
│       ├── benchmark.py                    # 알고리즘 비교 평가
│       ├── spawn_hunter_se.py              # 로봇 단독 스폰/검증
│       ├── drive_hunter_se.py              # Ackermann 수동 주행 검증
│       ├── algorithms/
│       │   ├── tqc/
│       │   │   ├── tqc_agent.py           #   Actor (Gaussian) + Quantile Critic
│       │   │   ├── tqc_trainer.py         #   Isaac Lab 환경 연동 학습 루프
│       │   │   └── networks.py            #   MLP Actor, QuantileCritic
│       │   ├── td7/
│       │   │   ├── td7_agent.py           #   SALE 인코더, 성능 회귀 복원
│       │   │   └── td7_trainer.py
│       │   ├── sac/
│       │   │   └── sac_agent.py
│       │   └── common/
│       │       ├── buffer.py              #   LAP Prioritized Experience Replay
│       │       └── logger.py             #   TensorBoard + JSON 로거
│       └── deploy/
│           ├── export_policy.py           #   ONNX / TorchScript 내보내기
│           ├── test_policy.py
│           └── ros2_node_template.py
│
├── apps/
│   ├── isaaclab.python.kit                # Isaac Lab GUI 실행 구성
│   └── isaaclab.python.headless.kit       # Isaac Lab 헤드리스 실행 구성
│
└── ros2/
    ├── autodrive_interfaces/              # [ROS2 패키지 1] 커스텀 인터페이스
    │   ├── srv/
    │   │   ├── GetAction.srv              #   float32[] state → float32[] action
    │   │   ├── LoadPolicy.srv             #   string model_path → bool success
    │   │   └── GetStatus.srv
    │   └── action/
    │       └── RunEpisode.action
    │
    └── isaaclab_ros2_bridge/              # [ROS2 패키지 2] 실로봇 배포 브리지
        ├── launch/
        │   └── deploy_hunter.launch.py
        ├── config/
        │   ├── policy.yaml                #   정책 파일 경로 및 입출력 설정
        │   └── robot.yaml                 #   토픽 매핑 (/odom, /scan, /cmd_vel)
        └── isaaclab_ros2_bridge/
            ├── policy_node.py             #   ONNX 정책 → /cmd_vel 발행
            ├── state_processor.py         #   /scan → 80-sector LiDAR 상태
            └── action_converter.py        #   2D 액션 → Ackermann 조향/속도
```

---

## 관측 / 행동 공간

| 항목 | 차원 | 내용 |
|---|---|---|
| 관측 (obs) | 82D | LiDAR 80 sector (min-pooling) + `[goal_dist, goal_angle]` |
| 행동 (act) | 2D | `linear_vel [-1, 1]`, `angular_vel [-1, 1]` |
| LiDAR 범위 | 5.0 m | 360° / 80 sector 균등 분할 |
| 맵 크기 | 16 × 16 m | 벽 + 랜덤 원통 장애물 N개 |

---

## 보상 함수

```
r_step = goal_progress(Δdist) × k_p          # 목표 접근 보상 (dense)
       + goal_reached × R_goal                # 목표 도달 보상 (sparse, +100)
       + collision × P_col                    # 충돌 페널티 (sparse, -10)
       + obstacle_proximity(min_lidar)        # 근접 페널티 (zone-based, smooth)
       + time_penalty                         # 생존 비용 (-0.01/step)

종료 조건:
  - goal_dist < 0.3 m               → goal reached
  - min_lidar_dist < 0.3 m          → collision
  - episode_steps >= max_steps       → timeout
```

---

## 지원 알고리즘

| 알고리즘 | 유형 | 특징 |
|---|---|---|
| TQC | Off-policy | Quantile 분산 RL, 안정적 수렴 — **주력** |
| TD7 | Off-policy | LAP 우선순위 버퍼, 고성능 |
| SAC | Off-policy | 엔트로피 정규화, 탐색 효율 우수 |
| PPO (RSL-RL) | On-policy | 빠른 학습, 대규모 병렬 환경 |

---

## 패키지 설치

```bash
# Isaac Lab 의존성 설치
cd /workspace/isaaclab
./isaaclab.sh -i

# 커스텀 패키지 설치
cd /workspace/hunter_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```

---

## 로봇 물리 검증

```bash
# 로봇 스폰 확인 (관절 구조 출력)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/spawn_hunter_se.py

# Ackermann 수동 주행 검증 (직진 → 좌회전 → 직진 → 우회전 → 정지)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_hunter_se.py --headless
```

---

## 학습 실행

> 모든 명령은 `/workspace/hunter_autodrive/` 디렉터리에서 실행합니다.

### TQC / TD7 학습 (`train_tqc.py`)

```bash
# TQC 학습 (기본)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc --num_envs 64 --headless

# TD7 학습 (LAP PER 기본 활성화)
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo td7 --num_envs 64 --headless

# 커스텀 config 지정
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc --cfg source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/agents/tqc_cfg.yaml \
    --headless
```

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--algo` | `tqc` | 알고리즘 선택 (`tqc` / `td7` / `sac`) |
| `--task` | — | 학습 태스크 ID |
| `--num_envs` | cfg 기본값 | 병렬 환경 수 |
| `--cfg` | 태스크 내 기본 YAML | 커스텀 config 파일 경로 |
| `--seed` | cfg 기본값 | 랜덤 시드 |
| `--headless` | `False` | 헤드리스 실행 |

로그 저장 경로: `logs/{algo}/lidar_nav/{timestamp}/`

---

### 시각화 (`play.py`)

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play.py \
    --task Isaac-LidarNav-Hunter-Play-v0 \
    --num_envs 4

# 체크포인트 직접 지정
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play.py \
    --task Isaac-LidarNav-Hunter-Play-v0 \
    --num_envs 4 \
    --checkpoint logs/tqc/lidar_nav/.../model_final.pt
```

---

### 알고리즘 비교 벤치마크 (`benchmark.py`)

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --tqc_ckpt logs/tqc/lidar_nav/.../model_final.pt \
    --td7_ckpt logs/td7/lidar_nav/.../model_final.pt \
    --num_envs 16 --eval_episodes 20
```

---

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

---

## 개발 단계

### Phase 1 — 기반 구축 ✅ 완료
- [x] `isaaclab_autodrive` 패키지 생성 및 `pip install -e` 등록
- [x] `hunter_se/` USD 물리 모델 구축 및 검증 (Physics.usda, hunter_se_cfg.py)
  - fr_left_joint 비대칭(localRot1 부호 반전) 수정 → 직진 좌편향 해결
  - 전륜 자유회전 damping 15 → 0.5 (진동 해결)
  - solver_velocity_iteration_count 16 → 4 (TGS 안정화)
- [x] `drive_hunter_se.py` 수동 주행 검증 (Ackermann, 시간 계산 수정)
- [x] 알고리즘 코어 이식: TQC / TD7 / SAC / buffer(LAP PER) / logger

### Phase 2 — path tracking 계열 구현 ✅ 완료 (legacy 보존)
- [x] `path_tracking_env.py` (crosstrack error 기반) → `legacy/`로 격리
- [x] `rough_terrain_tracking`, `hybrid_control`, `multi_track` 스켈레톤 → `legacy/`
- [x] RSL-RL PPO / TQC / TD7 연동 완료

### Phase 3 — LiDAR 자율주행 태스크 구성 🔄 진행 중
- [ ] **Phase A.** legacy 격리: path_tracking 계열 → `direct/legacy/`
- [ ] **Phase B.** `lidar_nav/` 골격 생성 (dummy obs로 환경 루프 동작 확인)
- [ ] **Phase C.** MVP 검증: LiDAR sensor + goal/obstacle 스폰 + reward/termination
- [ ] **Phase D.** TQC 학습 연결 (`Isaac-LidarNav-Hunter-v0`)
- [ ] **Phase E.** 논문 환경 재현 (80-sector, 16×16 맵, 랜덤화 완성)

### Phase 4 — 실로봇 배포
- [ ] ONNX 정책 내보내기 (`export_policy.py`)
- [ ] `isaaclab_ros2_bridge` 노드: `/scan` → 80-sector 상태 변환 → `/cmd_vel`
- [ ] Sim-to-Real 갭 분석

---

## 이식 출처

| 이식 대상 | 원본 출처 |
|---|---|
| `hunter_se/` | URDF → USD 변환 + 물리 파라미터 직접 수정 |
| `assets/robots/hunter.py` | `Hybrid_DRL_Deployments/.../hunter.py` (USD 경로 교체) |
| `algorithms/tqc/tqc_agent.py` | `drl_agent/scripts/policy/tqc_agent.py` |
| `algorithms/td7/td7_agent.py` | `drl_agent/scripts/policy/td7_agent.py` |
| `algorithms/common/buffer.py` | `drl_agent/scripts/utils/buffer.py` (LAP PER) |
| `ros2/autodrive_interfaces/` | `drl_agent_interfaces/` (확장) |
| `ros2/.../state_processor.py` | `drl_agent/scripts/environment/environment.py` (참고) |
| `utils/angle.py` | `Hybrid_DRL_Deployments/.../angle.py` |
