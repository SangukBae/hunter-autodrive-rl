# hunter_se_v0 — RL용 procedural Hunter SE 자산

`hunter_se_v0`는 이 저장소에서 RL 학습과 제어 검증에 사용하는 Hunter SE V0 자산입니다. `pxr` Python API로 USD articulation을 절차적으로 만들고, Hunter SE의 검증된 관절 위치/질량 분배를 그대로 가져옵니다.

핵심 목적은 다음 두 가지입니다.

- RL 환경에서 안정적으로 쓸 수 있는 단순한 물리 구조 제공
- 조향/구동 관절 이름 유지 (Ackermann 계산기와 호환)

시각 형상은 `/robot_isaac/ugv_gazebo_sim/hunter_se/hunter_se_description/meshes/` 의 STL 파일에서 빌드된 `hunter_se_v0_meshes.usdc`(binary)를 참조합니다. 충돌 형상(Box/Sphere)은 물리 전용으로 비가시 처리됩니다.

## 1. 파일 구성

| 파일 | 역할 | 비고 |
|---|---|---|
| `__init__.py` | 패키지 설명 | 현재는 설명용 모듈이며 별도 export는 없음 |
| `build_usd.py` | USD 생성기 | `pxr` API로 `hunter_se_v0.usda` + `hunter_se_v0_meshes.usdc` 생성 |
| `hunter_se_v0.usda` | 체크인된 생성 결과물 | Isaac Lab이 실제로 읽는 USD 자산 |
| `hunter_se_v0_meshes.usdc` | 시각 메시 바이너리 | STL에서 변환된 링크별 메시 (binary USDC) |
| `hunter_se_v0_cfg.py` | `ArticulationCfg` | USD 경로, 스폰 위치, actuator 튜닝 정의 |
| `ackermann.py` | Ackermann 변환기 | 중심 조향각/선속도를 좌우 조향각과 후륜 각속도로 변환 |

## 2. 현재 자산 구조

이 자산은 원본 `hunter_se/`보다 단순한 7-link / 6-joint 구조입니다.

### 링크 구성

- `base_link`
- `fr_steer_left_link`
- `fr_left_link`
- `fr_steer_right_link`
- `fr_right_link`
- `re_left_link`
- `re_right_link`

### 조인트 구성

- 전륜 조향: `fr_steer_left_joint`, `fr_steer_right_joint`
- 전륜 자유회전: `fr_left_joint`, `fr_right_joint`
- 후륜 구동: `re_left_joint`, `re_right_joint`

원본 `hunter_se/`에 있던 `front_steer_joint`, `rear_wheel_joint`는 V0에서 제거되었습니다.

### 계층 구조 특징

`build_usd.py`는 모든 rigid body 링크를 `/HunterSEV0`의 직접 자식으로 두는 플랫 계층을 사용합니다. 조인트 트리는 USD 계층이 아니라 `/HunterSEV0/Physics` 아래 joint의 `body0/body1` 참조로 정의됩니다. 이 구조는 PhysX의 nested rigid body 이슈를 피하기 위한 설계입니다.

## 3. 현재 코드 기준 주요 제원

### 기구/치수

| 항목 | 값 | 출처 |
|---|---:|---|
| 축거 (`WHEELBASE`) | `0.548 m` | `ackermann.py`, `build_usd.py` |
| 전륜 윤거 (`FRONT_TRACK`) | `0.492 m` | `ackermann.py` |
| 후륜 윤거 (`REAR_TRACK`) | `0.504 m` | `ackermann.py` |
| 바퀴 반지름 | `0.1375 m` | `ackermann.py`, `build_usd.py`, `hunter_se_v0.usda` |
| 바퀴 폭 | `0.080 m` | `build_usd.py` 설계값 |
| 최대 조향각 | `±22°` = `±0.384 rad` | `ackermann.py`, joint limit |
| 최고 속도 | `1.333 m/s` | `ackermann.py` |
| 차체 크기 | `0.817 × 0.640 × 0.120 m` | `build_usd.py` |
| 기본 스폰 높이 | `z = 0.2955 m` | `hunter_se_v0_cfg.py` |

### 질량

| 링크 그룹 | 질량 |
|---|---:|
| 차체 (`base_link`) | `23.106 kg` |
| 너클 2개 | `2 × 3.149 kg` |
| 바퀴 4개 | `4 × 3.149 kg` |
| 총 질량 | `42.0 kg` |

즉, 현재 V0 자산은 공식 매뉴얼 기준 `42 kg` 총질량에 맞춰 재배분된 단순화 모델입니다.

## 4. 충돌 형상과 시각 형상

현재 구현은 "기본 도형 기반"이지만, 정확히는 다음 조합입니다.

- 차체 충돌: `Cube` (비가시, 물리 전용)
- 너클 충돌: `Cube` (비가시, 물리 전용)
- 바퀴 충돌: `Sphere` (비가시, 물리 전용) — 접지 안정성을 위해 Sphere 선택
- 시각 형상: `hunter_se_v0_meshes.usdc` 의 STL 메시 (각 링크 `/visual` prim으로 참조)

STL 메시 출처: `/robot_isaac/ugv_gazebo_sim/hunter_se/hunter_se_description/meshes/`

## 5. 제어 구조와 actuator 설정

### 조인트 축 규약

- 조향 조인트 axis: `Z`
- 바퀴 조인트 axis: `Y`
- `ackermann.py` 부호 규약:
  - 양수 조향각 = 좌회전
  - 양수 선속도/바퀴 각속도 = 전진

### 런타임 actuator (`hunter_se_v0_cfg.py`)

| 그룹 | 대상 조인트 | 설정 |
|---|---|---|
| `wheels` | `re_left_joint`, `re_right_joint` | `DCMotorCfg(saturation_effort=30, effort_limit=11, velocity_limit=15, damping=15)` |
| `steering` | `fr_steer_left_joint`, `fr_steer_right_joint` | `ImplicitActuatorCfg(stiffness=500, damping=50, effort_limit_sim=50)` |
| `front_wheels` | `fr_left_joint`, `fr_right_joint` | `ImplicitActuatorCfg(stiffness=0, damping=0.01)` |

### 중요한 점

`hunter_se_v0.usda` 안에도 각 joint에 `PhysicsDriveAPI:angular`가 들어 있습니다. 하지만 실제 Isaac Lab 런타임 튜닝은 `hunter_se_v0_cfg.py`의 actuator 설정이 우선합니다.

예를 들어:

- 후륜 joint의 USD 기본 damping은 `17453`이지만, 런타임 `DCMotorCfg`는 `damping=15`를 사용합니다.
- 전륜 자유회전 joint의 USD 기본 damping은 `0.5`이지만, 런타임 actuator는 `0.01`을 사용합니다.
- 전륜 조향 joint의 USD 기본 강성은 매우 크지만, 런타임 actuator는 `500/50` 저강성 서보로 운용됩니다.

따라서 이 폴더를 이해할 때는 "USD는 제어 가능 상태를 보장하는 최소 drive 정의", "실제 제어 특성은 cfg가 결정"이라고 보는 것이 맞습니다.

## 6. Ackermann 변환

`ackermann.py`의 `HunterSEAckermann`는 다음 입력을 받습니다.

- 입력: 차체 중심 조향각 `delta_c [rad]`, 선속도 `v [m/s]`
- 출력: 좌/우 전륜 조향각, 좌/우 후륜 각속도

이 모듈은 다음 곳에서 직접 사용됩니다.

- `source/isaaclab_autodrive_tasks/.../lidar_nav/lidar_nav_env.py`

## 7. 저장소 내 실제 사용 위치

### 직접 import

```python
from hunter_se_v0.hunter_se_v0_cfg import HUNTER_SE_V0_CFG
from hunter_se_v0.ackermann import HunterSEAckermann
```

### 프로젝트 자산 레이어에서 재노출

`source/isaaclab_autodrive/isaaclab_autodrive/assets/robots/hunter.py`는 `HUNTER_SE_V0_CFG`를 다시 import해서 프로젝트 공용 자산 경로로 노출합니다.

### RL 환경 연결

`source/isaaclab_autodrive_tasks/isaaclab_autodrive_tasks/direct/lidar_nav/lidar_nav_env_cfg.py`는 기본 로봇으로 `HUNTER_SE_V0_CFG`를 사용합니다.

## 8. USD 생성과 재생성

### 자동 생성

`hunter_se_v0_cfg.py`는 `hunter_se_v0.usda`가 없으면 import 시 자동으로 `build_usd.py`를 호출합니다.

주의:

- 이 자동 생성은 `pxr`를 사용할 수 있는 Isaac Sim Python 환경에서만 동작합니다.
- 일반 시스템 Python에서 import만으로 생성되는 구조는 아닙니다.

### 수동 생성

```bash
/workspace/isaaclab/isaaclab.sh -p hunter_se_v0/build_usd.py
```

## 9. 현재 체크인된 USD와 생성기 동기화 상태

현재 저장소의 `hunter_se_v0.usda`는 핵심 물리 파라미터 기준으로는 `build_usd.py`와 대체로 맞습니다.

일치하는 예:

- 차체 질량 `23.106 kg`
- 바퀴 반지름 `0.1375 m`
- 후륜 joint 기본 damping `17453`
- 차체 `Cube` / 바퀴 `Sphere` 충돌 구조

다만 완전 동일하다고 보기는 어렵습니다.

- 체크인된 `hunter_se_v0.usda`에는 `base_link/Lidar` prim이 들어 있습니다.
- 현재 `build_usd.py`에는 그 RTX LiDAR prim을 생성하는 코드가 없습니다.

즉, `USD를 다시 생성하면 현재 체크인본과 100% 동일하지 않을 수 있습니다.`

다행히 현재 RL 태스크는 이 USD 내부의 RTX LiDAR를 직접 사용하지 않고, `lidar_nav_env.py`에서 `MultiMeshRayCaster`를 별도로 생성하므로 학습 파이프라인의 핵심 동작은 이 차이에 의존하지 않습니다.

## 10. 설계 특징

`hunter_se_v0`는 URDF→USD 변환본 대신 `pxr` API로 직접 생성한 procedural USD입니다.

| 항목 | 선택 이유 |
|---|---|
| 기본 도형(Box/Sphere) 충돌 | PhysX 안정성, 불필요한 메시 복잡도 제거 |
| virtual joints 없음 | RL용 단순화, 관절 수 최소화 |
| 바퀴 충돌: `Sphere` | 접지 안정성, 롤링 마찰 자연스러움 |
| 주요 관절 이름 유지 | 기존 Ackermann 계산기와 호환 |
