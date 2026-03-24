# td7 — TD7

TD7 off-policy 강화학습 알고리즘 구현입니다.

**논문:** Fujimoto & Gu, "TD7: Re-Establishing Baselines for Offline RL" (NeurIPS 2023)

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `td7_agent.py` | 네트워크 정의 + `TD7Agent` 학습 로직 |
| `td7_trainer.py` | `TD7Trainer` — Isaac Lab 환경 연동 학습 루프 |

---

## `td7_agent.py`

### 네트워크 클래스

| 클래스 | 구조 | 역할 |
|---|---|---|
| `Encoder` | Linear → ELU × 2 → Linear → LayerNorm | SALE: 상태를 잠재 임베딩으로 인코딩 |
| `TD3Actor` | Linear → ReLU × 2 → Linear → Tanh | 결정론적 정책 (행동 출력) |
| `TD7Critic` | (state + action + embedding) → Linear × 3 | 듀얼 Q 네트워크 (Q1, Q2 동시 출력) |

### `TD7Agent`

TQCAgent와 동일한 인터페이스를 제공합니다.

**TD7의 핵심 개선사항:**

1. **SALE (State-Action Learned Embeddings)**
   - Encoder가 상태를 잠재 임베딩으로 변환
   - Critic이 (상태, 행동, 임베딩) 모두를 입력으로 사용
   - 표현력 향상 + 학습 안정성 개선

2. **Checkpointing**
   - 일정 주기마다 현재 성능과 체크포인트 성능을 비교
   - 성능 회귀(regression) 감지 시 이전 체크포인트로 파라미터 복원
   - 학습 불안정성으로 인한 성능 저하 방지

3. **LAP PER (Latent Action Prioritized Experience Replay)**
   - 잠재 공간에서의 행동 거리를 우선순위로 사용
   - 중요한 전환을 더 자주 샘플링
   - `common/buffer.py`의 `LAP` 클래스와 연동

4. **Clipped Double Q-Learning (TD3 스타일)**
   - Q1, Q2 중 최솟값을 타깃으로 사용
   - 과대추정 억제

---

## `td7_trainer.py` — `TD7Trainer`

TQCTrainer와 동일한 학습 루프 구조를 가집니다. TD7Agent의 Checkpointing 기능을 활성화하기 위한 추가 로직이 포함됩니다.

### 로그 저장 경로

`logs/td7/{experiment_name}/{timestamp}/`
- `model_{step}.pt`: 중간 체크포인트
- `model_final.pt`: 최종 체크포인트
- `tensorboard/`: TensorBoard 로그
- `metrics.json`: 평가 지표 JSON
