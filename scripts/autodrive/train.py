# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE RSL-RL PPO 학습 스크립트.

사용 예시:
    # 기본 학습 (4096 환경, 헤드리스)
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train.py \
        --task Isaac-PathTracking-Hunter-v0 \
        --num_envs 4096 --headless

    # 소규모 디버그
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/train.py \
        --task Isaac-PathTracking-Hunter-v0 \
        --num_envs 64 --headless
"""

import argparse
import sys
import os

# Isaac Lab cli_args 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl")

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402 (Isaac Lab RSL-RL CLI 유틸)

parser = argparse.ArgumentParser(description="Hunter SE RSL-RL PPO 학습")
parser.add_argument("--task",           type=str,  default="Isaac-PathTracking-Hunter-v0")
parser.add_argument("--num_envs",       type=int,  default=None)
parser.add_argument("--seed",           type=int,  default=None)
parser.add_argument("--max_iterations", type=int,  default=None)
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,  default=200)
parser.add_argument("--video_interval", type=int,  default=2000)
parser.add_argument("--track",          type=str,  default="austin",
                    choices=["austin", "brandshatch", "silverstone"],
                    help="학습에 사용할 트랙 (기본값: austin)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import logging
from datetime import datetime
from importlib import metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_autodrive_tasks  # noqa: F401 — 환경 등록
from isaaclab_autodrive.terrains.track import TRACK_CSV_MAP

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg):
    """학습 메인 함수."""
    # CLI 인수로 설정 오버라이드
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.track_csv = TRACK_CSV_MAP[args_cli.track]
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 로그 디렉토리 설정
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    print(f"[INFO] 로그 경로: {log_dir}")
    print_dict(agent_cfg.to_dict())

    # 환경 생성
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # 비디오 래퍼
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # RSL-RL 벡터 환경 래퍼
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 체크포인트 경로 설정
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 러너 생성 및 학습
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO] 체크포인트 로드: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
