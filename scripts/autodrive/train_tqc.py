# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE TQC / TD7 오프-폴리시 학습 스크립트.

지원 알고리즘:
    tqc  — Truncated Quantile Critics (기본값)
    td7  — TD7 with SALE + Checkpointing + LAP PER

사용 예시:
    # TQC 학습
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
        --task Isaac-LidarNav-Hunter-v0 \
        --algo tqc --num_envs 64 --headless

    # TD7 학습 (LAP PER 활성화)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
        --task Isaac-LidarNav-Hunter-v0 \
        --algo td7 --num_envs 64 --headless

    # 커스텀 config 경로
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train_tqc.py \
        --task Isaac-LidarNav-Hunter-v0 \
        --algo tqc --cfg path/to/tqc_cfg.yaml --headless
"""

import argparse
import os
import sys


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hunter SE TQC/TD7 오프-폴리시 학습")
parser.add_argument("--task",          type=str,  default="Isaac-LidarNav-Hunter-v0")
parser.add_argument("--algo",          type=str,  default="tqc", choices=["tqc", "td7"])
parser.add_argument("--cfg",           type=str,  default=None,  help="YAML 설정 파일 경로 (없으면 태스크 내 기본값 사용)")
parser.add_argument("--num_envs",      type=int,  default=None)
parser.add_argument("--seed",          type=int,  default=None)
parser.add_argument("--lidar_use_rtx", action="store_true", default=False,
                    help="RTX OmniLidar 사용 (teleop 와 동일 방식). num_envs 1~8 권장.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import gymnasium as gym
import yaml
from datetime import datetime

import isaaclab_autodrive_tasks  # noqa: F401 — 환경 등록

# 알고리즘별 트레이너
sys.path.insert(0, os.path.dirname(__file__))
from algorithms.tqc import TQCTrainer
from algorithms.td7 import TD7Trainer


def _load_cfg(task: str, algo: str, cfg_path: str | None) -> dict:
    """YAML 설정 로드.

    cfg_path가 주어지면 해당 파일을 로드하고,
    없으면 태스크 패키지 내 agents/{algo}_cfg.yaml을 탐색합니다.
    """
    if cfg_path is not None:
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    # 태스크 entry_point에서 패키지 경로 추출
    spec = gym.spec(task)
    entry = spec.entry_point  # "isaaclab_autodrive_tasks.direct.path_tracking.path_tracking_env:..."
    pkg   = entry.split(":")[0].rsplit(".", 1)[0]           # "isaaclab_autodrive_tasks.direct.path_tracking"

    # Python 패키지 경로 → 파일 경로로 변환
    import importlib
    mod  = importlib.import_module(pkg)
    task_dir = os.path.dirname(mod.__file__)
    yaml_path = os.path.join(task_dir, "agents", f"{algo}_cfg.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {yaml_path}\n"
            f"--cfg 옵션으로 경로를 직접 지정하세요."
        )

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _verify_lidar(env, env_cfg, algo: str) -> None:
    """학습 시작 전 LiDAR 센서 동작 확인.

    env.reset() → 관측값의 LiDAR 구간 통계 출력 → 무작위 액션 1 step 후
    값 변화 여부를 확인한다. 이상 징후가 있으면 경고를 출력하고 계속 진행한다.
    (학습을 강제 중단하지 않음 — 사용자가 결과를 보고 판단)
    """
    import numpy as np
    import torch

    lidar_mode   = "RTX" if env_cfg.lidar_use_rtx else "Warp"
    num_sectors  = env_cfg.num_sectors   # 80
    lidar_range  = env_cfg.lidar_range   # 5.0 m
    action_dim   = env_cfg.action_space  # 2

    sep = "─" * 50
    print(f"\n[{algo.upper()}] LiDAR 동작 확인 ({lidar_mode} 모드)")
    print(sep)

    def _extract_lidar(obs_or_dict) -> torch.Tensor:
        """Isaac Lab 환경의 dict/tensor 관측에서 LiDAR 슬라이스를 추출한다."""
        if isinstance(obs_or_dict, dict):
            raw = obs_or_dict["policy"]
        else:
            raw = obs_or_dict
        if not isinstance(raw, torch.Tensor):
            raw = torch.tensor(np.asarray(raw), dtype=torch.float32)
        return raw[:, :num_sectors].float()

    try:
        # ── reset → 초기 관측 ────────────────────────────────────────────────
        obs, _ = env.reset()
        lidar_t = _extract_lidar(obs)

        n_envs       = lidar_t.shape[0]
        all_max_pct  = float((lidar_t >= 0.999).float().mean()) * 100.0
        mean_dist    = float(lidar_t.mean()) * lidar_range
        min_dist     = float(lidar_t.min())  * lidar_range

        print(f"  환경 수        : {n_envs}")
        print(f"  LiDAR sector   : {num_sectors}  (최대 범위 {lidar_range} m)")
        print(f"  평균 감지 거리 : {mean_dist:.2f} m")
        print(f"  최소 감지 거리 : {min_dist:.2f} m")
        print(f"  최대 범위 비율 : {all_max_pct:.1f}%  ← 100% 이면 장애물 미감지")

        # ── 무작위 액션 1 step → 값 변화 확인 ───────────────────────────────
        _obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        rng_action = torch.tensor(
            np.random.uniform(-1.0, 1.0, (n_envs, action_dim)).astype(np.float32),
            device=_obs_tensor.device,
        )
        obs2, _, _, _, _ = env.step(rng_action)
        lidar_t2 = _extract_lidar(obs2)

        delta = float((lidar_t2 - lidar_t).abs().mean())
        print(f"  1 step 후 변화 : {delta:.5f}  ← 0.00000 이면 센서 고정 의심")

        # ── 판정 ─────────────────────────────────────────────────────────────
        warnings = []
        if all_max_pct >= 99.9:
            warnings.append("모든 sector 가 최대 범위 — 장애물 없거나 센서 미동작")
        if delta == 0.0:
            warnings.append("step 후 LiDAR 값 무변화 — 센서 동결 가능성")

        if warnings:
            for w in warnings:
                print(f"  [경고] {w}")
            print(f"  결과: 이상 징후 감지 — 확인 후 학습 진행 권장")
        else:
            print(f"  결과: [OK] LiDAR 정상 동작")

    except Exception as exc:
        print(f"  [오류] 확인 중 예외 발생: {exc}")
        print(f"  RTX 모드라면 _init_rtx_lidars() 로그를 먼저 확인하세요.")

    print(sep + "\n")


def main():
    task  = args_cli.task
    algo  = args_cli.algo

    # ── 설정 로드 ─────────────────────────────────────────────────────────────
    cfg = _load_cfg(task, algo, args_cli.cfg)

    if args_cli.seed is not None:
        cfg["seed"] = args_cli.seed
    if args_cli.device is not None:
        cfg["device"] = args_cli.device

    # ── 환경 생성 ─────────────────────────────────────────────────────────────
    # track/num_envs 오버라이드를 위해 env_cfg 항상 직접 생성
    import importlib
    spec      = gym.spec(task)
    cfg_entry = spec.kwargs["env_cfg_entry_point"]    # "module.path:ClassName"
    mod_path, cls_name = cfg_entry.rsplit(":", 1)
    cfg_mod   = importlib.import_module(mod_path)
    env_cfg   = getattr(cfg_mod, cls_name)()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.lidar_use_rtx:
        env_cfg.lidar_use_rtx = True
        print(f"[{algo.upper()}] RTX LiDAR 모드 활성화 (teleop 동일 방식). "
              f"num_envs={env_cfg.scene.num_envs} — 느릴 수 있음, 1~8 권장.")

    env = gym.make(task, cfg=env_cfg)

    # ── LiDAR 동작 확인 ───────────────────────────────────────────────────────
    _verify_lidar(env, env_cfg, algo)

    # ── 로그 디렉토리 ─────────────────────────────────────────────────────────
    exp_name = cfg.get("experiment_name", f"hunter_{algo}")
    run_ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir  = os.path.abspath(os.path.join("logs", algo, exp_name, run_ts))

    print(f"[{algo.upper()}] task={task}, log_dir={log_dir}")

    # ── 학습 ─────────────────────────────────────────────────────────────────
    if algo == "tqc":
        trainer = TQCTrainer(env, cfg, log_dir)
    elif algo == "td7":
        trainer = TD7Trainer(env, cfg, log_dir)
    else:
        raise ValueError(f"알 수 없는 알고리즘: {algo}")

    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
