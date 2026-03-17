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
parser.add_argument("--task",     type=str, default="Isaac-LidarNav-Hunter-v0")
parser.add_argument("--algo",     type=str, default="tqc", choices=["tqc", "td7"])
parser.add_argument("--cfg",      type=str, default=None, help="YAML 설정 파일 경로 (없으면 태스크 내 기본값 사용)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed",     type=int, default=None)
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

    env = gym.make(task, cfg=env_cfg)

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
