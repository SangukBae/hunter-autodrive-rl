# scripts/autodrive — 학습 및 평가 스크립트

Isaac Lab 환경과 RL 알고리즘을 연결하는 실행 스크립트와 알고리즘 구현체를 담고 있습니다.

---

## 파일 목록

```
scripts/autodrive/
├── train_tqc.py            # TQC / TD7 학습 메인
├── play_lidar_nav.py       # 정책 평가 및 시각화
├── benchmark.py            # 알고리즘 비교 평가
├── spawn_hunter_se.py      # 로봇 스폰 검증
├── drive_hunter_se.py      # Ackermann 수동 주행 검증
├── test_lidar_nav.py       # 환경 단위 테스트
├── train.py                # [legacy] RSL-RL PPO 학습
├── play.py                 # [legacy] PPO 정책 평가
└── algorithms/
    ├── tqc/                # TQC 구현
    ├── td7/                # TD7 구현
    ├── sac/                # SAC 구현
    ├── common/             # 공통 버퍼, 로거
    └── deploy/             # [예정] ONNX/TorchScript 내보내기
```

---

## 주요 스크립트

### `train_tqc.py` — 학습 메인

TQC 또는 TD7 알고리즘으로 LiDAR 자율주행 정책을 학습합니다.

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --algo tqc --num_envs 64 --headless
```

**동작 순서:**
1. CLI 인수 파싱
2. YAML 하이퍼파라미터 로드 (`agents/tqc_cfg.yaml` 또는 `--cfg` 지정 경로)
3. Gymnasium 환경 생성
4. Trainer 초기화 (`TQCTrainer` 또는 `TD7Trainer`)
5. `trainer.train()` 호출 → 로그 저장 (`logs/{algo}/{name}/{timestamp}/`)

**인수:**

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--algo` | `tqc` | 알고리즘 (`tqc` / `td7`) |
| `--task` | `Isaac-LidarNav-Hunter-v0` | 환경 ID |
| `--num_envs` | cfg 기본값 | 병렬 환경 수 |
| `--cfg` | 태스크 내 기본 YAML | 커스텀 하이퍼파라미터 YAML |
| `--seed` | `0` | 랜덤 시드 |
| `--device` | `cuda:0` | 연산 디바이스 |
| `--headless` | `False` | GUI 없이 실행 |

---

### `play_lidar_nav.py` — 정책 평가/시각화

저장된 체크포인트를 로드하여 결정론적 정책으로 평가합니다.

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play_lidar_nav.py \
    --task Isaac-LidarNav-Hunter-Play-v0 \
    --algo tqc \
    --checkpoint logs/tqc/hunter_tqc/TIMESTAMP/model_final.pt \
    --num_envs 4
```

**출력 통계:**
- 목표 도달률 (goal_reached_rate)
- 충돌률 (collision_rate)
- 평균 에피소드 보상 (avg_reward)
- 평균 에피소드 스텝 (avg_steps)

**인수:**

| 인수 | 설명 |
|---|---|
| `--task` | 평가 환경 ID |
| `--algo` | 알고리즘 (`tqc` / `td7`) |
| `--checkpoint` | 체크포인트 `.pt` 파일 경로 |
| `--num_envs` | 병렬 환경 수 |
| `--eval_episodes` | 평가할 에피소드 수 |
| `--max_steps` | 에피소드당 최대 스텝 |
| `--headless` | 헤드리스 실행 |

---

### `benchmark.py` — 알고리즘 비교

여러 알고리즘 체크포인트를 동일 환경에서 비교 평가합니다.

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
    --task Isaac-LidarNav-Hunter-v0 \
    --tqc_ckpt logs/tqc/.../model_final.pt \
    --td7_ckpt logs/td7/.../model_final.pt \
    --num_envs 16 --eval_episodes 20
```

**출력:** 성공률, 충돌률, 보상, 스텝 비교 + JSON 리포트 저장

---

### `spawn_hunter_se.py` — 로봇 스폰 검증

로봇을 Isaac Sim에 스폰하고 관절/링크 정보를 출력합니다. 로봇 모델 구조를 확인하거나 물리 파라미터를 검증할 때 사용합니다.

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/spawn_hunter_se.py --headless
```

---

### `drive_hunter_se.py` — Ackermann 수동 주행 검증

미리 정의된 시나리오(직진 → 좌회전 → 직진 → 우회전 → 정지)로 로봇을 주행하며 Ackermann 제어기가 정상 작동하는지 확인합니다.

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/autodrive/drive_hunter_se.py --headless
```

기대 출력: 초당 속도/위치 로그, 최종 위치

---

### `test_lidar_nav.py` — 환경 단위 테스트

환경 임포트, 관측 형상, 보상 값 등 환경 기본 동작을 헤드리스로 검증합니다.

---

## algorithms/ 하위 패키지

자세한 내용은 `algorithms/README.md`를 참조하세요.

| 폴더 | 내용 |
|---|---|
| `tqc/` | TQC 네트워크, 에이전트, 트레이너 |
| `td7/` | TD7 에이전트(SALE + LAP PER), 트레이너 |
| `sac/` | SAC 에이전트 |
| `common/` | LAP 리플레이 버퍼, TensorBoard 로거 |
| `deploy/` | [예정] ONNX/TorchScript 정책 내보내기 |
