# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""학습된 PPO 정책 시각화 스크립트.

사용 예시:
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/play.py \
        --task Isaac-PathTracking-Hunter-Play-v0 \
        --num_envs 16 \
        --checkpoint logs/rsl_rl/hunter_path_tracking/2024-01-01_00-00-00/model_1000.pt
"""

import argparse
import sys
import os

sys.path.insert(0, "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl")

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Hunter SE 정책 시각화")
parser.add_argument("--task",       type=str, default="Isaac-PathTracking-Hunter-Play-v0")
parser.add_argument("--num_envs",   type=int, default=16)
parser.add_argument("--video",      action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--track",      type=str, default="austin",
                    choices=["austin", "brandshatch", "silverstone"],
                    help="시각화에 사용할 트랙 (기본값: austin)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import gymnasium as gym
import torch

from importlib import metadata

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import isaaclab_autodrive_tasks  # noqa: F401
from isaaclab_autodrive.terrains.track import TRACK_CSV_MAP


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg):
    """시각화 메인 함수."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.track_csv = TRACK_CSV_MAP[args_cli.track]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 체크포인트 경로
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] 체크포인트 로드: {resume_path}")

    # 환경 생성
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="videos/play",
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # 추론 루프
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
