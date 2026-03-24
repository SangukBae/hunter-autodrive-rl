# algorithms — RL 알고리즘 구현체

TQC, TD7 off-policy 알고리즘과 공통 유틸리티를 포함합니다. 모든 알고리즘은 Isaac Lab 환경과 `LAP` 리플레이 버퍼를 공통으로 사용합니다.

---

## 폴더 구조

```
algorithms/
├── tqc/
│   ├── networks.py         # Actor, QuantileCritic 네트워크
│   ├── tqc_agent.py        # TQC 에이전트
│   └── tqc_trainer.py      # Isaac Lab 연동 학습 루프
├── td7/
│   ├── td7_agent.py        # TD7 에이전트 (네트워크 포함)
│   └── td7_trainer.py      # TD7 학습 루프
└── common/
    ├── buffer.py           # LAP 우선순위 경험 리플레이
    └── logger.py           # TensorBoard + JSON 로거
```

---

## tqc/ — Truncated Quantile Critics

**논문:** "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (Kuznetsov et al., ICML 2020)

### `networks.py`

| 클래스 | 역할 |
|---|---|
| `Actor` | Gaussian 정책 네트워크. 상태 → (mean, log_std) → tanh 샘플링. ReLU 활성화, 2-layer MLP. |
| `QuantileCritic` | N개의 Critic 앙상블. 각 Critic이 Q 분포를 M개의 분위수로 출력. ELU 활성화. |

### `tqc_agent.py` — `TQCAgent`

| 메서드 | 설명 |
|---|---|
| `__init__()` | Actor, n_critics×QuantileCritic, 타깃 네트워크, 엔트로피 계수 초기화 |
| `select_action(state, deterministic)` | 탐색(확률적) 또는 평가(결정론적) 행동 반환 |
| `train(replay_buffer)` | 배치 샘플 → critic/actor/entropy 업데이트 → 타깃 소프트 업데이트 |
| `save(path)` / `load(path)` | 체크포인트 저장/로드 |

**핵심 특징:**
- 상위 분위수 제거(`top_quantiles_to_drop_per_net × n_critics`)로 Q 과대추정 억제
- 자동 엔트로피 조정 (`ent_coef="auto_1.0"`)
- LAP 우선순위 버퍼 연동

### `tqc_trainer.py` — `TQCTrainer`

| 메서드 | 설명 |
|---|---|
| `__init__(env, cfg)` | YAML cfg로 에이전트/버퍼/로거 초기화 |
| `train()` | 전체 학습 루프: warmup → collect → update → eval → checkpoint |
| `_collect_transitions()` | num_envs 병렬 환경에서 전환 수집 → 버퍼 저장 |
| `_update_agent()` | 미니배치로 에이전트 1회 업데이트 |
| `_evaluate()` | 결정론적 정책으로 eval_episodes 평가 |

---

## td7/ — TD7

**논문:** "TD7: Re-Establishing Baselines for Offline RL" (Fujimoto & Gu, NeurIPS 2023)

### `td7_agent.py` — 네트워크 + `TD7Agent`

| 클래스 | 역할 |
|---|---|
| `Encoder` | SALE 인코더. 상태 → 잠재 임베딩 (LayerNorm 출력). ELU 활성화. |
| `TD3Actor` | 결정론적 정책. 상태 → 행동 (Tanh). ReLU 활성화. |
| `TD7Critic` | 듀얼 Q 네트워크. (상태, 행동, 잠재 임베딩) → Q값 2개. |
| `TD7Agent` | 위 네트워크 통합 + 학습 로직 |

**핵심 특징:**
- **SALE**: Encoder가 상태를 잠재 임베딩으로 변환, Critic 표현력 향상
- **Checkpointing**: 성능 회귀 감지 시 이전 체크포인트로 파라미터 복원
- **LAP PER**: 잠재 공간 행동 거리 기반 우선순위 샘플링
- **Clipped Double Q-Learning**: Q1, Q2 최솟값을 타깃으로 사용

### `td7_trainer.py` — `TD7Trainer`

TQCTrainer와 동일한 인터페이스. TD7Agent의 Checkpointing 기능 활성화 로직 포함.

---

## common/ — 공통 유틸리티

### `buffer.py` — `LAP`

모든 off-policy 알고리즘이 공유하는 경험 리플레이 버퍼입니다.

| 메서드 | 설명 |
|---|---|
| `add(state, action, next_state, reward, done)` | 전환 저장 (순환 버퍼) |
| `sample()` | 균일 또는 우선순위 가중 배치 샘플링 |
| `update_priority(priority)` | TD 오차 기반 우선순위 갱신 |

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `max_size` | 1,000,000 | 최대 저장 전환 수 |
| `batch_size` | 256 | 샘플 배치 크기 |
| `prioritized` | `False` | 우선순위 샘플링 활성화 |

### `logger.py` — `Logger`

TensorBoard와 JSON 파일에 동시에 메트릭을 기록합니다.

| 메서드 | 설명 |
|---|---|
| `log_scalar(tag, value, step)` | 단일 스칼라 기록 |
| `log_dict(metrics, step)` | 여러 메트릭 일괄 기록 |
| `flush()` | JSON 즉시 저장 |
| `close()` | TensorBoard writer 종료 |

로그는 `logs/{algo}/{experiment_name}/{timestamp}/` 하위에 저장됩니다.
