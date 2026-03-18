# hunter_se — 로봇 에셋

Hunter SE 모바일 로봇의 USD 3D 모델과 Isaac Lab `ArticulationCfg`를 담고 있습니다. 원본은 URDF 기반이며 `urdf-usd-converter v0.1.0`으로 변환된 후 Isaac Lab 연동을 위해 물리 파라미터가 수동으로 조정되었습니다.

## 1. 폴더 구성

- `hunter_se_description.usda`
  - 메인 진입점 asset interface.
  - `Payload/Contents.usda`를 payload로 불러온다.
- `hunter_se_cfg.py`
  - Isaac Lab에서 사용하는 `ArticulationCfg`.
  - 스폰 설정, articulation solver 설정, actuator 구성을 정의한다.
- `Payload/Contents.usda`
  - Physics, Geometry, Materials 레이어를 합치는 payload 조립 파일.
- `Payload/Physics.usda`
  - 조인트, rigid body 질량/관성, collision approximation 등 물리 설정.
- `Payload/Geometry.usda`
  - 링크 배치, 시각 메쉬 참조, 일부 간단한 기본 형상.
- `Payload/Materials.usda`
  - 시각 재질 정의.
- `Payload/GeometryLibrary.usdc`
  - 실제 시각 메쉬 라이브러리.
- `Payload/MaterialsLibrary.usdc`
  - 시각 재질 라이브러리.

## 2. 좌표계와 단위

- 단위계
  - `metersPerUnit = 1`
  - `kilogramsPerUnit = 1`
- 업축
  - `upAxis = "Z"`
- Isaac Lab 초기 스폰 위치
  - `init_state.pos = (0.0, 0.0, 0.3)`

즉, 길이는 미터, 질량은 킬로그램, 위쪽 방향은 +Z 기준이다.

## 3. 로봇 크기와 주요 치수

### 3.1 USD 전체 바운딩 박스 추정

`hunter_se_description.usda`의 `extentsHint` 기준 대략:

- 길이: 약 `0.820 m`
- 너비: 약 `0.635 m`
- 높이: 약 `0.295 m`

이 값은 asset 힌트용 extents이며, 실제 주행/접촉 해석에서는 각 링크의 collision 형상이 더 중요하다.

### 3.2 조인트 배치로부터 읽을 수 있는 치수

`Payload/Physics.usda`의 joint anchor 기준:

- 전륜 좌 조향축 위치: `(0.34058, 0.24619, -0.1535)`
- 전륜 우 조향축 위치: `(0.34219, -0.24619, -0.1535)`
- 후륜 좌 휠축 위치: `(-0.2078, 0.252, -0.158)`
- 후륜 우 휠축 위치: `(-0.2078, -0.252, -0.158)`

이 값으로부터 추정 가능한 차체 치수:

- 축거 wheelbase: 약 `0.549 m`
- 전륜 윤거 front track: 약 `0.492 m`
- 후륜 윤거 rear track: `0.504 m`

### 3.3 문서/주석에 기록된 제원

`hunter_se_cfg.py`의 설명 주석 기준:

- 조향 방식: Ackermann 전륜 조향
- 최대 조향각: `±22°` = `±0.384 rad`
- 최고 속도: `4.8 km/h` = 약 `1.333 m/s`
- 자체 중량: `42 kg`
- 지상고: `120 mm`

이 항목들은 Python 모듈의 설명 텍스트에 적힌 정보이며, 일부는 실제 동역학 파라미터가 아니라 로봇 스펙 설명이다.

## 4. 조인트 구성과 작동 방식

이 자산은 총 8개 조인트를 사용한다.

### 4.1 후륜 구동

- `re_left_joint`
- `re_right_joint`

특징:

- 둘 다 revolute joint.
- `velocity mode`로 사용하도록 설계됨.
- 구동축은 USD상 `physics:axis = "Z"`로 정의되어 있지만, local rotation이 걸려 있어 실제 구름 방향과 맞춰진다.

### 4.2 전륜 조향

- `fr_steer_left_joint`
- `fr_steer_right_joint`

특징:

- 둘 다 revolute joint.
- `position mode`로 사용.
- 조향 제한:
  - `lowerLimit = -22.0`
  - `upperLimit = 22.0`

즉, 실제 조향 입력은 좌우 독립 조향각으로 들어가며 Ackermann 기하에 맞춘 좌우 각도를 외부 제어기에서 계산해 넣는 구조다.

### 4.3 전륜 자유 회전

- `fr_left_joint`
- `fr_right_joint`

특징:

- 직접 구동하지 않는 자유 회전 휠 역할.
- actuator는 연결되어 있지만 `stiffness = 0`이라 구동용이 아니라 낮은 damping만 가진 자유 회전에 가깝다.

### 4.4 가상 보조 조인트

- `front_steer_joint`
- `rear_wheel_joint`

특징:

- 실제 구동/조향 조인트라기보다 구조적 연결이나 보조 링크 역할.
- 매우 큰 stiffness/damping으로 사실상 잠금(lock) 상태로 운용된다.

## 5. 액추에이터 설정

`hunter_se_cfg.py` 기준 actuator 구성은 다음과 같다.

### 5.1 wheels

- 대상 조인트:
  - `re_left_joint`
  - `re_right_joint`
- 설정:
  - `stiffness = 0.0`
  - `damping = 17453.0`

해석:

- 후륜을 속도 제어용으로 쓰기 위한 설정이다.
- `stiffness = 0` 이므로 위치 스프링이 없고, 큰 damping 값으로 속도 추종 특성을 만든다.

### 5.2 steering

- 대상 조인트:
  - `fr_steer_left_joint`
  - `fr_steer_right_joint`
- 설정:
  - `stiffness = 1e7`
  - `damping = 1e5`
  - `effort_limit_sim = 6000.0`

해석:

- 전륜 조향을 강한 위치 제어로 잡아주는 설정이다.
- 응답성이 높고 목표 조향각을 빠르게 추종한다.

### 5.3 virtual_joints

- 대상 조인트:
  - `front_steer_joint`
  - `rear_wheel_joint`
- 설정:
  - `stiffness = 1e7`
  - `damping = 1e5`
  - `effort_limit_sim = 1e6`

해석:

- 가상 링크를 사실상 고정시키기 위한 잠금용 actuator다.

### 5.4 front_wheels

- 대상 조인트:
  - `fr_left_joint`
  - `fr_right_joint`
- 설정:
  - `stiffness = 0.0`
  - `damping = 0.5`

해석:

- 전륜 자유 회전용 설정.
- 원래 URDF friction 값 `15`가 점성 damping처럼 변환되면 과도한 제동이 생기기 때문에, 현재는 `0.5` 수준으로 낮춰져 있다.

## 6. 조인트별 핵심 물리 파라미터

### 6.1 조향 조인트

- `fr_steer_left_joint`
  - limit: `[-22°, +22°]`
  - stiffness: `1e7`
  - damping: `1e5`
  - maxForce: `6000`
- `fr_steer_right_joint`
  - limit: `[-22°, +22°]`
  - stiffness: `1e7`
  - damping: `1e5`
  - maxForce: `6000`

### 6.2 후륜 구동 조인트

- `re_left_joint`
  - stiffness: `0`
  - damping: `17453`
  - targetVelocity 초기값: `0`
- `re_right_joint`
  - stiffness: `0`
  - damping: `17453`
  - targetVelocity 초기값: `0`

### 6.3 전륜 자유 회전 조인트

- `fr_left_joint`
  - stiffness: `0`
  - damping: `0.5`
- `fr_right_joint`
  - stiffness: `0`
  - damping: `0.5`

### 6.4 가상 조인트

- `front_steer_joint`
  - limit: 약 `[-41.25°, +41.25°]`
  - stiffness: `1e7`
  - damping: `1e5`
- `rear_wheel_joint`
  - stiffness: `1e7`
  - damping: `1e5`

## 7. 질량, 무게중심, 관성

### 7.1 링크별 질량

- `base_link`: `24.73 kg`
- `fr_steer_left_link`: `3.149 kg`
- `fr_left_link`: `3.149 kg`
- `fr_steer_right_link`: `3.149 kg`
- `fr_right_link`: `3.149 kg`
- `re_left_link`: `3.149 kg`
- `re_right_link`: `3.149 kg`
- `front_steer_link`: `0.0049179 kg`
- `rear_wheel_link`: `0.0049179 kg`

합산 질량은 약 `43.634 kg` 이다.

즉, 폴더 안 자산의 실제 물리 질량 총합은 주석에 적힌 `42 kg`와 거의 같은 수준이다.

### 7.2 주요 무게중심

- `base_link` center of mass:
  - `(0.037414003, -0.0003730052, -0.07712829)`
- `re_left_link` center of mass:
  - `(1.245e-9, 0.0000017252, -0.010284)`
- `re_right_link` center of mass:
  - `(0, 0, -0.010284)`

전륜 및 조향 링크도 각각 독립적인 질량과 관성을 가진 rigid body로 설정되어 있다.

### 7.3 주요 관성

- `base_link diagonalInertia`
  - `(0.12309484, 0.21914472, 0.31372702)`
- 바퀴/조향 링크 diagonalInertia
  - 대부분 `(0.0222, 0.0222, 0.0378075)`

## 8. 충돌 형상과 접촉 해석

### 8.1 차체

- `base_link` collision approximation: `convexHull`

### 8.2 바퀴

- `fr_left_link`, `fr_right_link`, `re_left_link`, `re_right_link` collision approximation:
  - `boundingSphere`

의미:

- 시각 메쉬를 그대로 triangle mesh 충돌로 쓰는 대신 구형 근사 충돌을 사용한다.
- 계산은 가볍지만, 바퀴 접지 반경이나 횡방향 접촉 특성을 정밀하게 반영하는 데는 불리할 수 있다.

### 8.3 가상 링크

- `front_steer_link` 와 `rear_wheel_link` 에는 매우 작은 cylinder 형상이 들어 있다.
- 해당 형상 크기:
  - `radius = 0.005 m`
  - `height = 0.001 m`

이 링크들은 실제 타이어 역할보다는 보조 구조에 가깝다.

## 9. Isaac Lab 스폰/아티큘레이션 설정

`hunter_se_cfg.py` 기준:

- contact sensors 활성화: `True`
- gravity 비활성화 여부: `False`
- gyroscopic forces: `True`
- max depenetration velocity: `1.0`
- self collisions: `False`
- solver position iterations: `32`
- solver velocity iterations: `16`
- sleep threshold: `0.005`
- stabilization threshold: `0.001`
- `copy_from_source = False`

## 10. 파일만 보고 알 수 있는 로봇 운용 방식 요약

이 로봇은 구조적으로 다음 방식으로 운용된다.

- 후륜 2개를 속도 제어해서 추진력을 만든다.
- 전륜 2개는 좌우 독립 조향 조인트로 방향을 만든다.
- 전륜 바퀴 자체는 자유롭게 회전한다.
- 중앙의 `front_steer_joint`, `rear_wheel_joint` 는 실제 구동용이 아니라 구조 보조용으로 잠가 둔다.

즉, `후륜 구동 + 전륜 Ackermann 조향` 구조다.

## 11. 주의할 점

- 이 폴더에는 실제 구동 제어기 로직은 없다.
  - Ackermann 각도 계산, 목표 속도 계산, 주행 시나리오 정의는 바깥 스크립트가 담당한다.
- `Geometry.usda` 안의 Gazebo 스타일 `mu1`, `mu2`, `kp`, `kd` 텍스트 블록은 설명용 메타데이터에 가깝다.
  - 실제 PhysX material 바인딩과 완전히 같은 의미는 아니다.
- 최고 속도, 최대 조향각, 중량 같은 일부 값은 Python 파일의 설명 주석과 USD의 실제 파라미터가 함께 존재하므로, 운영 시에는 둘을 구분해서 보는 것이 좋다.

## 12. 한 줄 요약

Hunter SE 자산은 Isaac Lab에서 사용할 수 있는 `후륜 속도 구동 + 전륜 위치 조향` 구조의 UGV 모델이며, 총 8개 조인트, 약 43.6kg 물리 질량, 약 0.55m 축거, 약 0.50m 후륜 윤거를 가진 Ackermann 차량 자산으로 구성되어 있다.
