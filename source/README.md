# source — Python 패키지

Isaac Lab 연동을 위한 두 개의 Python 패키지를 포함합니다. 두 패키지 모두 `pip install -e`로 설치해야 합니다.

---

## 패키지 구성

```
source/
├── isaaclab_autodrive/          # 공통 모듈 (로봇 에셋, 유틸리티)
└── isaaclab_autodrive_tasks/    # RL 환경 태스크 (Gymnasium 등록)
```

---

## isaaclab_autodrive

**역할:** 로봇 에셋 설정과 공통 유틸리티를 제공하는 기반 패키지입니다.

```
isaaclab_autodrive/
├── assets/
│   └── robots/
│       └── hunter.py           # HUNTER_SE_CFG — hunter_se/ 참조 래퍼
└── utils/
    ├── angle.py                 # 각도 정규화 유틸 (angle_mod, rot_mat_2d)
    ├── cubic_spline.py          # [legacy] 3차 스플라인 경로 보간
    └── lqr_controller.py       # [legacy] LQR 경로 추종 제어기
```

## isaaclab_autodrive_tasks

**역할:** Gymnasium 환경을 등록하는 태스크 패키지입니다. `import isaaclab_autodrive_tasks` 만으로 4개 환경이 자동 등록됩니다.

```
isaaclab_autodrive_tasks/
└── direct/
    ├── lidar_nav/               # ★ 주력 — LiDAR 자율주행 태스크
    └── legacy/                  # 보존용 — path tracking 계열 환경
```

자세한 내용은 각 패키지 디렉터리의 README를 참조하세요.

---

## 설치

```bash
cd /workspace/hunter_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_autodrive_tasks
```
