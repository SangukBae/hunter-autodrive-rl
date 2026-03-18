# isaaclab_autodrive_tasks

Gymnasium 환경을 Isaac Lab에 등록하는 태스크 패키지입니다. 패키지를 임포트하면 4개의 LiDAR 자율주행 환경이 자동으로 등록됩니다.

---

## 파일 목록

```
isaaclab_autodrive_tasks/
├── setup.py
└── isaaclab_autodrive_tasks/
    ├── __init__.py
    └── direct/
        ├── __init__.py                 # lidar_nav 임포트 (legacy 제외)
        ├── lidar_nav/                  # ★ 주력 태스크
        │   ├── __init__.py             # 4개 환경 Gymnasium 등록
        │   ├── lidar_nav_env_cfg.py    # 환경 설정 클래스 4종
        │   ├── lidar_nav_env.py        # 환경 본체
        │   └── agents/
        │       ├── tqc_cfg.yaml        # TQC 하이퍼파라미터
        │       └── td7_cfg.yaml        # TD7 하이퍼파라미터
        └── legacy/                     # 보존용 (자동 임포트 없음)
            ├── path_tracking/
            ├── rough_terrain_tracking/
            ├── hybrid_control/
            └── multi_track/
```

---

## 등록 환경

| 환경 ID | Phase | num_envs | 용도 |
|---|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | C | 64 | 학습 (해석적 LiDAR) |
| `Isaac-LidarNav-Hunter-Play-v0` | C | 4 | 시각화/평가 |
| `Isaac-LidarNav-Hunter-PhaseE-v0` | E | 64 | 학습 (물리 벽 + 물리 장애물) |
| `Isaac-LidarNav-Hunter-PhaseE-Play-v0` | E | 4 | PhaseE 시각화/평가 |

---

## 각 파일 역할

### `direct/__init__.py`

`lidar_nav` 패키지만 임포트합니다. `legacy/`는 자동 임포트에서 제외되어 있습니다.

### `direct/lidar_nav/__init__.py`

`gymnasium.register()`를 호출해 4개 환경을 등록합니다. Phase C와 Phase E 각각 학습용(64 envs)과 평가용(4 envs)을 등록합니다.

### `direct/lidar_nav/lidar_nav_env_cfg.py`

환경 설정 데이터클래스 4종을 정의합니다.

| 클래스 | 상속 | 특징 |
|---|---|---|
| `LidarNavEnvCfg` | `DirectRLEnvCfg` | Phase C 기본 (64 envs, walls=False) |
| `LidarNavEnvCfgPlay` | `LidarNavEnvCfg` | Phase C 평가용 (4 envs) |
| `LidarNavEnvCfgPhaseE` | `LidarNavEnvCfg` | Phase E (walls=True, physical_obstacles=True) |
| `LidarNavEnvCfgPhaseEPlay` | `LidarNavEnvCfgPhaseE` | Phase E 평가용 (4 envs) |

주요 파라미터:

| 파라미터 | 값 | 설명 |
|---|---|---|
| `observation_space` | 82 | 80 LiDAR + goal_dist + goal_angle |
| `action_space` | 2 | linear_vel, angular_vel |
| `lidar_range` | 5.0 m | 최대 감지 거리 |
| `num_sectors` | 80 | 360° / 4.5° 간격 |
| `map_size` | ±8.0 m | 16×16m 맵 |
| `num_obstacles` | 5 | 원통 장애물 수 |
| `goal_reward` | 100.0 | 목표 도달 보상 |
| `collision_penalty` | -10.0 | 충돌 페널티 |
| `sim.dt` | 1/200 s | 물리 시뮬레이션 주기 |
| `decimation` | 4 | 제어 주기 = 0.02 s |

### `direct/lidar_nav/lidar_nav_env.py`

환경 본체입니다. 자세한 내용은 `lidar_nav/README.md`를 참조하세요.

### `direct/lidar_nav/agents/tqc_cfg.yaml`

TQC 하이퍼파라미터를 flat key YAML 형식으로 정의합니다.

```yaml
n_critics: 5
n_quantiles: 25
top_quantiles_to_drop_per_net: 2
actor_lr: 3.0e-4
discount: 0.99
buffer_size: 1000000
total_timesteps: 1000000
warmup_steps: 10000
```

### `direct/lidar_nav/agents/td7_cfg.yaml`

TD7 하이퍼파라미터를 flat key YAML 형식으로 정의합니다. TQC와 동일한 키 구조를 사용하며 TD7 전용 파라미터(LAP PER 등)가 포함됩니다.

### `direct/legacy/`

Phase 2에서 구현된 path tracking 계열 환경을 보존합니다. 자동 임포트에서 제외되어 있으며 현재 사용되지 않습니다.

| 서브폴더 | 내용 |
|---|---|
| `path_tracking/` | crosstrack error 기반 경로 추종 환경 |
| `rough_terrain_tracking/` | 거친 지형 경로 추종 환경 |
| `hybrid_control/` | 하이브리드 제어 환경 스켈레톤 |
| `multi_track/` | 멀티 트랙 환경 스켈레톤 |

---

## 설치

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```

## 환경 등록 확인

```bash
/workspace/isaaclab/isaaclab.sh -p - <<'EOF'
import isaaclab_autodrive_tasks
import gymnasium as gym
envs = [k for k in gym.envs.registry.keys() if "LidarNav" in k]
for e in sorted(envs): print(e)
EOF
```
