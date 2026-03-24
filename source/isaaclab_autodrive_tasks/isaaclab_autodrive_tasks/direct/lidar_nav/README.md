# lidar_nav — LiDAR 자율주행 태스크

Hunter SE V0 로봇의 RTX LiDAR 기반 goal-reaching + obstacle avoidance 환경입니다. 16×16m 물리 벽+장애물 맵에서 매 에피소드 랜덤 배치로 학습합니다.

---

## 파일 목록

```
lidar_nav/
├── __init__.py             # Gymnasium 환경 2개 등록
├── lidar_nav_env_cfg.py    # 환경 설정 데이터클래스 2종
├── lidar_nav_env.py        # 환경 본체 (LidarNavEnv)
└── agents/
    ├── __init__.py
    ├── tqc_cfg.yaml        # TQC 하이퍼파라미터
    └── td7_cfg.yaml        # TD7 하이퍼파라미터
```

---

## `__init__.py` — 환경 등록

`gymnasium.register()`로 2개 환경을 등록합니다.

| 환경 ID | 설정 클래스 | num_envs |
|---|---|---|
| `Isaac-LidarNav-Hunter-v0` | `LidarNavEnvCfg` | 64 |
| `Isaac-LidarNav-Hunter-Play-v0` | `LidarNavEnvCfgPlay` | 4 |

---

## `lidar_nav_env_cfg.py` — 환경 설정

`DirectRLEnvCfg`를 상속하는 2개의 데이터클래스입니다.

### 핵심 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `observation_space` | 82 | 80 LiDAR + goal_dist + goal_angle |
| `action_space` | 2 | linear_vel, angular_vel |
| `lidar_model` | `"os1-32"` | RTX OmniLidar 모델 |
| `lidar_use_rtx` | `True` | RTX OmniLidar 활성화 |
| `lidar_range` | 120.0 m | OS1 실물 스펙 최대 감지 거리 |
| `num_sectors` | 80 | 360° / 4.5° 간격 |
| `map_size` | ±8.0 m | 16×16m 맵 |
| `num_obstacles` | 10 | 매 에피소드 랜덤 배치 장애물 수 |
| `use_walls` | `True` | 물리 벽 경계 |
| `use_physical_obstacles` | `True` | 물리 장애물 실린더 |
| `lidar_collision_threshold` | 0.3 m | LiDAR 충돌 판정 임계값 |
| `goal_threshold` | 0.3 m | 목표 도달 판정 거리 |
| `max_linear_vel` | 1.0 m/s | 행동 스케일 |
| `max_angular_vel` | 1.0 rad/s | 행동 스케일 |
| `episode_length_s` | 60.0 s | 최대 에피소드 길이 |
| `sim.dt` | 1/200 s | 물리 시뮬레이션 주기 |
| `decimation` | 4 | 제어 주기 = 0.02 s |

---

## `lidar_nav_env.py` — 환경 본체

`DirectRLEnv`를 상속하는 `LidarNavEnv` 클래스입니다.

### 메서드 목록

| 메서드 | 설명 |
|---|---|
| `_setup_scene()` | 로봇, 바닥, 조명, 물리 벽(4면), 물리 장애물 스폰 |
| `_spawn_walls_per_env()` | 맵 경계에 static cuboid 벽 4개 스폰 |
| `_pre_physics_step(action)` | 행동 클리핑 → Ackermann 조향각 변환 → 관절 타깃 계산 |
| `_apply_action()` | 조향 위치 + 바퀴 속도를 관절에 적용 |
| `_get_observations()` | 82D 관측 계산 + `_last_lidar_obs` 캐시 갱신 |
| `_compute_lidar()` | RTX OmniLidar 부분 스캔 → 롤링-min → 80-sector 정규화 맵 |
| `_get_rewards()` | 다중 요소 보상 계산 |
| `_get_dones()` | 종료 조건 계산 (goal / LiDAR 충돌 / timeout) |
| `_reset_idx(env_ids)` | 로봇/목표/장애물 랜덤 리셋, RTX 버퍼 초기화 |
| `_get_collision_mask()` | `_last_lidar_obs.min() * lidar_range < lidar_collision_threshold` |
| `_get_min_obstacle_dist()` | LiDAR 최소 감지 거리 [m] (근접 페널티용) |

### 관측 공간 (82D)

```
obs[0:80]   → LiDAR 80 sector, 거리 정규화 [0, 1]
              (값 1.0 = 감지 없음, lidar_range 이상)
obs[80]     → goal_dist / map_size, [0, 1]
obs[81]     → goal_angle / π, [-1, 1] (로봇 로컬 프레임 기준)
```

### 행동 공간 (2D)

```
act[0] → linear_vel  [-1, 1] → [0, max_linear_vel] m/s  (전진만, 음수는 0 클리핑)
act[1] → angular_vel [-1, 1] → [-max_angular_vel, +max_angular_vel] rad/s
```

### 보상 함수

```
r = goal_progress × 5.0          # Δgoal_dist > 0 이면 양수 (dense)
  + 100.0  (if goal_reached)      # 목표 도달 보너스 (sparse)
  − 10.0   (if collision)         # LiDAR 충돌 페널티 (sparse)
  − proximity_penalty              # LiDAR 근접 smooth 페널티 (최대 2.0)
  − 0.01                          # time penalty per step
```

`proximity_penalty = proximity_penalty_max × clamp(1 − min_lidar_dist / proximity_threshold, 0, 1)`

### 종료 조건

| 조건 | 타입 | 설명 |
|---|---|---|
| `goal_dist < 0.3 m` | terminated | 목표 도달 (success) |
| LiDAR 최소 거리 < 0.3 m | terminated | 충돌 (failure) |
| `episode_steps >= max_steps` | truncated | 타임아웃 |

### RTX LiDAR 스캔 누적

OS1-32 @ 10Hz → 1회전 = 20 physics step ≈ 5 RL step.
매 RL step에서 수신된 부분 스캔(~72°)을 `_rtx_sector_buf`에 rolling-min 누적.
버퍼는 에피소드 리셋(`_reset_idx`) 시에만 `lidar_range`로 초기화됨.

```python
# 롤링 버퍼 업데이트 핵심 로직
self._rtx_sector_buf[i] = torch.minimum(
    self._rtx_sector_buf[i],
    torch.from_numpy(sector_dists).to(self.device),
)
lidar_obs = self._rtx_sector_buf / lidar_range   # 정규화 [0, 1]
```

### LiDAR 기반 충돌 판정

```python
# _get_observations()에서 매 스텝 캐시
self._last_lidar_obs = lidar_obs  # (num_envs, 80)

# 충돌 판정
min_dist_m = self._last_lidar_obs.min(dim=1).values * self.cfg.lidar_range
collision  = min_dist_m < self.cfg.lidar_collision_threshold
```

벽·장애물 모두 RTX 물리 메시를 통해 통합 판정됩니다.

### Ackermann 조향 변환

`hunter_se_v0/ackermann.py`의 `HunterSEAckermann`를 사용합니다.
angular_vel을 Ackermann 기하로 내/외륜 조향각으로 분배하고, 조향 변화율을 `±0.5 rad/s`로 제한합니다.

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
| `ent_coef` | `"auto_1.0"` | 자동 엔트로피 계수 |
| `buffer_size` | 1,000,000 | 리플레이 버퍼 크기 |
| `batch_size` | 256 | 미니배치 크기 |
| `total_timesteps` | 1,000,000 | 총 학습 스텝 |
| `warmup_steps` | 10,000 | 랜덤 수집 초기 스텝 |
| `eval_interval` | 10,000 | 평가 주기 |

## `agents/td7_cfg.yaml` — TD7 하이퍼파라미터

TQC와 동일한 flat key 구조입니다. TD7 전용 파라미터(`prioritized`, LAP PER 관련)가 포함됩니다.
