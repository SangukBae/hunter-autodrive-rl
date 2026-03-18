# common — 공통 유틸리티

모든 off-policy 알고리즘(TQC, TD7, SAC)이 공유하는 리플레이 버퍼와 로거입니다.

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `buffer.py` | `LAP` — 우선순위 경험 리플레이 버퍼 |
| `logger.py` | `Logger` — TensorBoard + JSON 통합 로거 |

---

## `buffer.py` — `LAP`

LAP(Latent Action Priority) 우선순위 경험 리플레이 버퍼입니다. `prioritized=False`로 설정하면 일반 균일 리플레이 버퍼로도 동작합니다.

### 생성 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `state_dim` | — | 상태 차원 (82) |
| `action_dim` | — | 행동 차원 (2) |
| `device` | — | PyTorch 디바이스 |
| `max_size` | 1,000,000 | 최대 전환 저장 수 |
| `batch_size` | 256 | 샘플 배치 크기 |
| `prioritized` | `False` | 우선순위 샘플링 활성화 여부 |

### 주요 메서드

| 메서드 | 설명 |
|---|---|
| `add(state, action, next_state, reward, done)` | 전환 저장. 버퍼가 가득 차면 가장 오래된 전환 덮어쓰기. |
| `sample()` | 배치 샘플링. `prioritized=True`면 우선순위 가중 샘플링. 반환: `(state, action, next_state, reward, not_done)` 텐서 튜플. |
| `update_priority(priority)` | 에이전트 업데이트 후 TD 오차 기반 우선순위 갱신. `prioritized=True`일 때만 유효. |

### 내부 구조

- numpy 배열로 전환 저장 (메모리 효율)
- `sample()` 호출 시 PyTorch 텐서로 변환하여 반환
- 순환 포인터(`ptr`)로 FIFO 덮어쓰기

---

## `logger.py` — `Logger`

학습 메트릭을 TensorBoard SummaryWriter와 JSON 파일에 동시 기록합니다.

### 생성 파라미터

| 파라미터 | 설명 |
|---|---|
| `log_dir` | 로그 저장 디렉터리 (`logs/{algo}/{name}/{timestamp}/`) |
| `experiment_name` | 실험 이름 (TensorBoard 표시용) |

### 주요 메서드

| 메서드 | 설명 |
|---|---|
| `log_scalar(tag, value, step)` | 단일 스칼라 값 기록 |
| `log_dict(metrics_dict, step)` | 딕셔너리의 모든 키-값 일괄 기록 |
| `flush()` | JSON 파일 즉시 디스크에 쓰기 |
| `close()` | TensorBoard writer 닫기 |

### 저장 파일

```
logs/{algo}/{experiment_name}/{timestamp}/
├── tensorboard/        # TensorBoard 이벤트 파일
└── metrics.json        # 평가 지표 JSON (step별 기록)
```

### 기록되는 주요 메트릭

| 태그 | 설명 |
|---|---|
| `train/critic_loss` | Critic 손실 |
| `train/actor_loss` | Actor 손실 |
| `train/ent_coef` | 현재 엔트로피 계수 값 |
| `eval/goal_reached_rate` | 목표 도달률 |
| `eval/collision_rate` | 충돌률 |
| `eval/avg_reward` | 평균 에피소드 보상 |
| `eval/avg_steps` | 평균 에피소드 스텝 수 |
