# tqc — Truncated Quantile Critics

TQC off-policy 강화학습 알고리즘 구현입니다.

**논문:** Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (ICML 2020)

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `networks.py` | `Actor`, `QuantileCritic` PyTorch 네트워크 |
| `tqc_agent.py` | `TQCAgent` — 학습 로직 및 행동 선택 |
| `tqc_trainer.py` | `TQCTrainer` — Isaac Lab 환경 연동 학습 루프 |

---

## `networks.py`

### `Actor`

Gaussian 정책 네트워크입니다.

```
입력: 상태 (82D)
      ↓ Linear(82, hdim) → ReLU
      ↓ Linear(hdim, hdim) → ReLU
출력: mean (action_dim), log_std (action_dim)
```

- `get_action()`: reparameterization trick으로 행동 샘플링 + log_prob 반환
- `forward()`: (mean, log_std) 반환 (결정론적 평가 시 사용)

### `QuantileCritic`

N개의 Critic 앙상블로 Q 분포를 분위수로 표현합니다.

```
입력: (상태, 행동)
      ↓ 각 Critic: Linear → ELU → Linear → ELU → Linear
출력: (batch, n_critics, n_quantiles)
```

---

## `tqc_agent.py` — `TQCAgent`

### 초기화 파라미터 (tqc_cfg.yaml과 대응)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `n_critics` | 5 | Critic 개수 |
| `n_quantiles` | 25 | 분위수 개수 |
| `top_quantiles_to_drop_per_net` | 2 | Critic당 제거할 상위 분위수 수 |
| `discount` | 0.99 | 할인 인수 γ |
| `tau` | 0.005 | 타깃 네트워크 소프트 업데이트 비율 |
| `ent_coef` | `"auto_1.0"` | 엔트로피 계수 (자동 조정) |

### `train(replay_buffer)` 업데이트 순서

1. **Critic 타깃 계산**: 다음 상태에서 행동 샘플 → 타깃 분위수 계산 → 상위 분위수 제거 → 벨만 타깃
2. **Critic 업데이트**: Huber Quantile Loss
3. **Actor 업데이트**: 엔트로피 보너스 포함 Q 최대화
4. **엔트로피 계수 업데이트**: `target_entropy = -action_dim`을 목표로 자동 조정
5. **타깃 네트워크 소프트 업데이트**: `τ × θ + (1-τ) × θ_target`
6. **LAP 우선순위 갱신**: `prioritized=True`인 경우 TD 오차 기반 업데이트

---

## `tqc_trainer.py` — `TQCTrainer`

Isaac Lab 환경과 TQCAgent를 연결하는 학습 루프입니다.

### 학습 흐름

```
초기화: env 생성 → agent 생성 → LAP 버퍼 생성 → Logger 생성
         ↓
warmup_steps: 랜덤 행동으로 버퍼 채우기
         ↓
반복 (total_timesteps):
  1. _collect_transitions(): num_envs 병렬 스텝 수행 → 버퍼 저장
  2. _update_agent(): 미니배치 샘플 → train() 호출
  3. eval_interval마다: _evaluate() → 체크포인트 저장
         ↓
완료: model_final.pt 저장
```

### 로그 저장 경로

`logs/tqc/{experiment_name}/{timestamp}/`
- `model_{step}.pt`: 중간 체크포인트
- `model_final.pt`: 최종 체크포인트
- `tensorboard/`: TensorBoard 로그
- `metrics.json`: 평가 지표 JSON
