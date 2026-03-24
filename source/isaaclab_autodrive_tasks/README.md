# isaaclab_autodrive_tasks

Gymnasium 환경을 Isaac Lab에 등록하는 태스크 패키지입니다. 패키지를 임포트하면 2개의 LiDAR 자율주행 환경이 자동으로 등록됩니다.

---

## 파일 목록

```
isaaclab_autodrive_tasks/
├── setup.py
└── isaaclab_autodrive_tasks/
    ├── __init__.py
    └── direct/
        ├── __init__.py
        └── lidar_nav/
            ├── __init__.py             # 2개 환경 Gymnasium 등록
            ├── lidar_nav_env_cfg.py    # 환경 설정 클래스 2종
            ├── lidar_nav_env.py        # 환경 본체
            └── agents/
                ├── tqc_cfg.yaml        # TQC 하이퍼파라미터
                └── td7_cfg.yaml        # TD7 하이퍼파라미터
```

---

## 등록 환경

| 환경 ID | num_envs | 용도 |
|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | 64 | 학습 (RTX LiDAR, 벽+장애물) |
| `Isaac-LidarNav-Hunter-Play-v0` | 4 | 시각화/평가 |

---

## 각 파일 역할

### `direct/lidar_nav/__init__.py`

`gymnasium.register()`를 호출해 2개 환경을 등록합니다.

### `direct/lidar_nav/lidar_nav_env_cfg.py`

환경 설정 데이터클래스 2종을 정의합니다.

| 클래스 | 상속 | 특징 |
|---|---|---|
| `LidarNavEnvCfg` | `DirectRLEnvCfg` | 학습용 기본 설정 (64 envs) |
| `LidarNavEnvCfgPlay` | `LidarNavEnvCfg` | 시각화/평가용 (4 envs, lidar_debug_vis=True) |

주요 파라미터:

| 파라미터 | 값 | 설명 |
|---|---|---|
| `observation_space` | 82 | 80 LiDAR + goal_dist + goal_angle |
| `action_space` | 2 | linear_vel, angular_vel |
| `lidar_model` | `"os1-32"` | RTX OmniLidar 모델 |
| `lidar_use_rtx` | `True` | RTX LiDAR 활성화 |
| `lidar_range` | 120.0 m | OS1 실물 스펙 최대 감지 거리 |
| `num_sectors` | 80 | 360° / 4.5° 간격 |
| `map_size` | ±8.0 m | 16×16m 맵 |
| `num_obstacles` | 10 | 매 에피소드 랜덤 배치 장애물 수 |
| `use_walls` | `True` | 물리 벽 경계 활성화 |
| `lidar_collision_threshold` | 0.3 m | LiDAR 최소 거리 충돌 판정 임계값 |
| `goal_reward` | 100.0 | 목표 도달 보상 |
| `collision_penalty` | -10.0 | 충돌 페널티 |
| `sim.dt` | 1/200 s | 물리 시뮬레이션 주기 |
| `decimation` | 4 | 제어 주기 = 0.02 s |

### `direct/lidar_nav/lidar_nav_env.py`

환경 본체입니다. 자세한 내용은 `lidar_nav/README.md`를 참조하세요.

### `direct/lidar_nav/agents/tqc_cfg.yaml`

TQC 하이퍼파라미터를 flat key YAML 형식으로 정의합니다.

### `direct/lidar_nav/agents/td7_cfg.yaml`

TD7 하이퍼파라미터를 flat key YAML 형식으로 정의합니다. TD7 전용 파라미터(LAP PER 등)가 포함됩니다.

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
