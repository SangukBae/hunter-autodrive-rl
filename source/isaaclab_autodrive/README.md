# isaaclab_autodrive

로봇 에셋 설정을 제공하는 기반 패키지입니다. `isaaclab_autodrive_tasks`에서 이 패키지를 의존성으로 사용합니다.

---

## 파일 목록

```
isaaclab_autodrive/
├── setup.py
└── isaaclab_autodrive/
    ├── __init__.py
    └── assets/
        └── robots/
            ├── __init__.py
            └── hunter.py       # HUNTER_SE_V0_CFG 임포트 래퍼
```

---

## `assets/robots/hunter.py`

`hunter_se_v0/hunter_se_v0_cfg.py`에 정의된 `HUNTER_SE_V0_CFG`를 임포트해서 재노출하는 래퍼 모듈입니다.

```python
from isaaclab_autodrive.assets.robots.hunter import HUNTER_SE_V0_CFG
```

환경 설정 클래스(`lidar_nav_env_cfg.py`)에서 이 경로를 통해 로봇 설정을 참조합니다.

---

## 설치

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
```
