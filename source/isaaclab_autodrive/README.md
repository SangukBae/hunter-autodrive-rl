# isaaclab_autodrive

로봇 에셋 설정과 공통 유틸리티를 제공하는 기반 패키지입니다. `isaaclab_autodrive_tasks`에서 이 패키지를 의존성으로 사용합니다.

---

## 파일 목록

```
isaaclab_autodrive/
├── setup.py
└── isaaclab_autodrive/
    ├── __init__.py
    ├── assets/
    │   └── robots/
    │       ├── __init__.py
    │       └── hunter.py           # HUNTER_SE_CFG 임포트 래퍼
    └── utils/
        ├── __init__.py
        ├── angle.py                # 각도 정규화 함수
        ├── cubic_spline.py         # [legacy] 3차 스플라인
        └── lqr_controller.py      # [legacy] LQR 제어기
```

---

## 각 파일 역할

### `assets/robots/hunter.py`

`hunter_se/hunter_se_cfg.py`에 정의된 `HUNTER_SE_CFG`를 임포트해서 재노출하는 래퍼 모듈입니다.

```python
from isaaclab_autodrive.assets.robots.hunter import HUNTER_SE_CFG
```

환경 설정 클래스에서 이 경로를 통해 로봇 설정을 참조합니다.

---

### `utils/angle.py`

각도 계산 유틸리티 함수를 제공합니다.

| 함수 | 설명 |
|---|---|
| `angle_mod(x, zero_2_2pi, degree)` | 각도를 `[-π, π)` 또는 `[0, 2π)` 범위로 정규화 |
| `rot_mat_2d(angle)` | 2D 회전 행렬 생성 |

`lidar_nav_env.py`의 goal_angle 계산에서 사용됩니다.

---

### `utils/cubic_spline.py` (legacy)

path tracking 환경에서 사용하던 경로 보간 모듈입니다. 현재 `lidar_nav` 태스크에서는 사용되지 않습니다.

| 클래스 / 함수 | 설명 |
|---|---|
| `CubicSpline1D` | 자연 3차 스플라인 보간 (1D) |
| `CubicSpline2D` | 호장 파라미터화 2D 경로 스플라인 |
| `calc_spline_course(x, y, ds)` | 웨이포인트로부터 보간 경로 생성 |

---

### `utils/lqr_controller.py` (legacy)

crosstrack/heading error 기반 LQR 경로 추종 제어기입니다. 현재 `lidar_nav` 태스크에서는 사용되지 않습니다.

| 클래스 | 설명 |
|---|---|
| `State` | 차량 상태 (x, y, yaw, v) |
| `LQRController` | DARE 솔버 기반 이산 LQR 제어기 |

---

## 설치

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
```
