# lidar_nav — LiDAR 자율주행 태스크

Hunter SE 로봇의 LiDAR 기반 goal-reaching + obstacle avoidance 환경입니다. Phase C(해석적 LiDAR)와 Phase E(물리 벽 + 물리 장애물)를 하나의 클래스로 통합 구현합니다.

---

## 파일 목록

```
lidar_nav/
├── __init__.py             # Gymnasium 환경 4개 등록
├── lidar_nav_env_cfg.py    # 환경 설정 데이터클래스 4종
├── lidar_nav_env.py        # 환경 본체 (LidarNavEnv)
└── agents/
    ├── __init__.py
    ├── tqc_cfg.yaml        # TQC 하이퍼파라미터
    └── td7_cfg.yaml        # TD7 하이퍼파라미터
```

---

## `__init__.py` — 환경 등록

`gymnasium.register()`로 4개 환경을 등록합니다.

| 환경 ID | 설정 클래스 | num_envs |
|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | `LidarNavEnvCfg` | 64 |
| `Isaac-LidarNav-Hunter-Play-v0` | `LidarNavEnvCfgPlay` | 4 |
| `Isaac-LidarNav-Hunter-PhaseE-v0` | `LidarNavEnvCfgPhaseE` | 64 |
| `Isaac-LidarNav-Hunter-PhaseE-Play-v0` | `LidarNavEnvCfgPhaseEPlay` | 4 |

---

## `lidar_nav_env_cfg.py` — 환경 설정

`DirectRLEnvCfg`를 상속하는 4개의 데이터클래스입니다.

### Phase 구분 파라미터

| 파라미터 | Phase C | Phase E |
|---|---|---|
| `use_walls` | `False` | `True` |
| `use_physical_obstacles` | `False` | `True` |
| `obstacle_radius_min/max` | 0.15 ~ 0.5 m | 0.3 m (고정) |

### 공통 핵심 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `observation_space` | 82 | 80 LiDAR + goal_dist + goal_angle |
| `action_space` | 2 | linear_vel, angular_vel |
| `lidar_range` | 5.0 m | 최대 LiDAR 감지 거리 |
| `num_sectors` | 80 | 360° / 4.5° 간격 |
| `map_size` | ±8.0 m | 16×16 m 맵 |
| `num_obstacles` | 5 | 원통 장애물 개수 |
| `goal_threshold` | 0.3 m | 목표 도달 판정 거리 |
| `max_linear_vel` | 1.0 m/s | 행동 스케일 |
| `max_angular_vel` | 1.0 rad/s | 행동 스케일 |
| `episode_length_s` | 60.0 s | 최대 에피소드 길이 |
| `sim.dt` | 1/200 s | 물리 시뮬레이션 주기 |
| `decimation` | 4 | 제어 주기 = 0.02 s |

---

## `lidar_nav_env.py` — 환경 본체

`DirectRLEnv`를 상속하는 `LidarNavEnv` 클래스입니다.

### 주요 물리 상수

```python
_WHEELBASE    = 0.548 m
_REAR_TRACK   = 0.504 m
_WHEEL_RADIUS = 0.129 m
_MAX_STEER    = 0.384 rad
_ROBOT_RADIUS = 0.30 m   # 충돌 판정용 근사 반경
```

### 메서드 목록

| 메서드 | 설명 |
|---|---|
| `_setup_scene()` | 로봇, 바닥, 조명, 장애물(Phase E), 벽(Phase E) 스폰 |
| `_spawn_walls_per_env()` | 맵 경계에 static cuboid 벽 4개 스폰 (Phase E) |
| `_pre_physics_step(action)` | 행동 클리핑 → Ackermann 조향각 변환 → 관절 타깃 계산 |
| `_apply_action()` | 조향 위치 + 바퀴 속도를 관절에 적용 |
| `_get_observations()` | 82D 관측 계산 (LiDAR 80 + goal_dist + goal_angle) |
| `_compute_lidar()` | GPU 벡터화 해석적 LiDAR (ray-cylinder + ray-AABB) |
| `_get_rewards()` | 다중 요소 보상 계산 |
| `_get_dones()` | 종료 조건 계산 (goal/collision/OOB/timeout) |
| `_reset_idx(env_ids)` | 선택된 환경의 로봇/목표/장애물 랜덤 리셋 |
| `_get_collision_mask()` | 원-원 충돌 감지 (Phase C) |
| `_get_min_obstacle_dist()` | 로봇 중심에서 가장 가까운 장애물 표면까지 거리 |

### 관측 공간 (82D)

```
obs[0:80]   → LiDAR 80 sector, 거리 정규화 [0, 1]
              (값 1.0 = 감지 없음, lidar_range 이상)
obs[80]     → goal_dist / map_size, [0, 1]
obs[81]     → goal_angle / π, [-1, 1] (로봇 로컬 프레임 기준)
```

### 행동 공간 (2D)

```
act[0] → linear_vel  [-1, 1] → [0, max_linear_vel] m/s  (전진만)
act[1] → angular_vel [-1, 1] → [-max_angular_vel, +max_angular_vel] rad/s
```

음수 linear_vel은 0으로 클리핑 (후진 없음).

### 보상 함수

```
r = goal_progress × 5.0          # Δgoal_dist > 0 이면 양수 (dense)
  + 100.0  (if goal_reached)      # 목표 도달 보너스 (sparse)
  − 10.0   (if collision)         # 충돌 페널티 (sparse)
  − proximity_penalty              # 장애물 근접 smooth 페널티 (최대 2.0)
  − 0.01                          # time penalty per step
```

`proximity_penalty = proximity_penalty_max × clamp(1 − min_dist / proximity_threshold, 0, 1)`

### 종료 조건

| 조건 | 타입 | 설명 |
|---|---|---|
| `goal_dist < 0.3 m` | terminated | 목표 도달 (success) |
| `robot-obstacle dist < ROBOT_RADIUS + obs_radius` | terminated | 충돌 (failure) |
| `|x| > map_size or |y| > map_size` | terminated | 맵 이탈 (Phase C only) |
| `episode_steps >= max_steps` | truncated | 타임아웃 |

### 해석적 LiDAR (`_compute_lidar`)

물리 센서 없이 GPU 텐서 연산으로 LiDAR를 계산합니다.

- **Phase C**: ray-cylinder 교차 계산 (장애물만)
- **Phase E**: ray-cylinder(장애물) + ray-AABB(벽) 동시 계산, 최솟값 취득

모든 연산은 `(num_envs, num_sectors)` 텐서로 벡터화되어 있어 GPU에서 병렬 실행됩니다.

### Ackermann 조향 변환 (`_pre_physics_step`)

angular_vel을 Ackermann 기하로 내/외륜 조향각으로 분배합니다.

```python
# 단순화된 공식
delta = arctan(WHEELBASE * angular_vel / linear_vel)
delta_left  = arctan(WHEELBASE / (WHEELBASE/tan(delta) - REAR_TRACK/2))
delta_right = arctan(WHEELBASE / (WHEELBASE/tan(delta) + REAR_TRACK/2))
```

---

## `agents/tqc_cfg.yaml` — TQC 하이퍼파라미터

`train_tqc.py`의 `TQCTrainer`가 읽는 flat key YAML입니다.

| 키 | 값 | 설명 |
|---|---|---|
| `n_critics` | 5 | Quantile Critic 개수 |
| `n_quantiles` | 25 | 분위수 개수 |
| `top_quantiles_to_drop_per_net` | 2 | 과대추정 억제용 상위 분위수 제거 수 |
| `actor_lr` / `critic_lr` | 3e-4 | 학습률 |
| `discount` | 0.99 | 할인 인수 |
| `tau` | 0.005 | 소프트 타깃 업데이트 비율 |
| `ent_coef` | `"auto_1.0"` | 자동 엔트로피 계수 (초기값 1.0) |
| `buffer_size` | 1,000,000 | 리플레이 버퍼 크기 |
| `batch_size` | 256 | 미니배치 크기 |
| `total_timesteps` | 1,000,000 | 총 학습 스텝 |
| `warmup_steps` | 10,000 | 랜덤 수집 초기 스텝 |
| `eval_interval` | 10,000 | 평가 주기 |

## `agents/td7_cfg.yaml` — TD7 하이퍼파라미터

TQC와 동일한 flat key 구조입니다. TD7 전용 파라미터(`prioritized`, LAP PER 관련)가 포함됩니다.
