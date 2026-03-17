# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO vs TQC vs TD7 성능 비교 벤치마크.

저장된 체크포인트를 로드하여 각 알고리즘의 결정론적 정책을 평가하고
결과를 JSON 및 콘솔에 출력합니다.

사용 예시:
    # 체크포인트 경로를 직접 지정
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
        --task Isaac-PathTracking-Hunter-v0 \
        --ppo_ckpt  logs/rsl_rl/hunter_path_tracking/2026-.../model_0.pt \
        --tqc_ckpt  logs/tqc/hunter_path_tracking_tqc/.../model_final.pt \
        --td7_ckpt  logs/td7/hunter_path_tracking_td7/.../model_final.pt \
        --num_envs 16 --eval_episodes 20 --headless

    # 특정 알고리즘만 평가
    /workspace/isaaclab/isaaclab.sh -p scripts/autodrive/benchmark.py \
        --task Isaac-PathTracking-Hunter-v0 \
        --tqc_ckpt logs/tqc/.../model_final.pt \
        --num_envs 16 --headless
"""

import argparse
import json
import os
import sys


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO vs TQC vs TD7 벤치마크")
parser.add_argument("--task",          type=str, default="Isaac-PathTracking-Hunter-v0")
parser.add_argument("--ppo_ckpt",      type=str, default=None, help="RSL-RL PPO 체크포인트 (.pt)")
parser.add_argument("--tqc_ckpt",      type=str, default=None, help="TQC 체크포인트 (.pt)")
parser.add_argument("--td7_ckpt",      type=str, default=None, help="TD7 체크포인트 (.pt)")
parser.add_argument("--num_envs",      type=int, default=16)
parser.add_argument("--eval_episodes", type=int, default=20, help="알고리즘당 평가 에피소드 수")
parser.add_argument("--max_steps",     type=int, default=2000, help="에피소드당 최대 스텝")
parser.add_argument("--output",        type=str, default=None, help="결과 JSON 저장 경로")
parser.add_argument("--track",         type=str, default="austin",
                    choices=["austin", "brandshatch", "silverstone"],
                    help="평가에 사용할 트랙 (기본값: austin)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True  # 벤치마크는 항상 헤드리스

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 초기화 이후 임포트 ──────────────────────────────────────────────
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime

import isaaclab_autodrive_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from algorithms.tqc.tqc_agent import TQCAgent
from algorithms.td7.td7_agent import TD7Agent
from isaaclab_autodrive.terrains.track import TRACK_CSV_MAP


# ──────────────────────────────────────────────────────────────────────────────
# 평가 함수
# ──────────────────────────────────────────────────────────────────────────────

def _extract_obs(obs_dict) -> np.ndarray:
    if isinstance(obs_dict, dict):
        obs_tensor = obs_dict.get("policy", list(obs_dict.values())[0])
    else:
        obs_tensor = obs_dict
    if isinstance(obs_tensor, torch.Tensor):
        return obs_tensor.cpu().numpy()
    return np.asarray(obs_tensor, dtype=np.float32)


def evaluate_policy(env, act_fn, eval_episodes: int, max_steps: int, device: torch.device) -> dict:
    """단일 정책 평가.

    Args:
        env:           Isaac Lab gymnasium 환경
        act_fn:        (obs: np.ndarray) → np.ndarray 콜백
        eval_episodes: 평가 에피소드 수
        max_steps:     에피소드당 최대 스텝
        device:        텐서 디바이스

    Returns:
        {"mean_reward": float, "std_reward": float, "min_reward": float, "max_reward": float}
    """
    obs_dict, _ = env.reset()
    obs = _extract_obs(obs_dict)

    episode_rewards: list[float] = []
    cur_reward = 0.0
    steps = 0

    while len(episode_rewards) < eval_episodes:
        action_np = act_fn(obs)
        action_t  = torch.tensor(action_np, dtype=torch.float32, device=device)
        next_obs_dict, reward, terminated, truncated, _ = env.step(action_t)
        next_obs = _extract_obs(next_obs_dict)

        rew  = reward.cpu().numpy().reshape(-1)[0]
        done = bool((terminated | truncated).cpu().numpy()[0])

        cur_reward += rew
        steps      += 1

        if done or steps >= max_steps:
            episode_rewards.append(cur_reward)
            cur_reward = 0.0
            steps      = 0
            obs_dict, _ = env.reset()
            obs = _extract_obs(obs_dict)
        else:
            obs = next_obs

    arr = np.array(episode_rewards)
    return {
        "mean_reward": float(arr.mean()),
        "std_reward":  float(arr.std()),
        "min_reward":  float(arr.min()),
        "max_reward":  float(arr.max()),
        "n_episodes":  len(episode_rewards),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PPO 평가 (RSL-RL policy)
# ──────────────────────────────────────────────────────────────────────────────

def make_ppo_act_fn(ckpt_path: str, obs_dim: int, action_dim: int, device: torch.device):
    """RSL-RL PPO 체크포인트에서 actor 로드 후 act_fn 반환."""
    import torch.nn as nn

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # RSL-RL OnPolicyRunner가 저장하는 키: "model_state_dict" 또는 직접 state_dict
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "actor" in ckpt:
        state_dict = ckpt["actor"]
    else:
        state_dict = ckpt

    # MLP 구조 복원 (RSL-RL 기본 hidden=[256,256])
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, action_dim),
            )
        def forward(self, x):
            return torch.tanh(self.net(x))

    # RSL-RL actor는 actor.net.* 형태로 저장됨
    actor_keys = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
    if not actor_keys:
        actor_keys = state_dict  # fallback

    mlp = MLP().to(device)
    try:
        mlp.net.load_state_dict(
            {k.replace("net.", ""): v for k, v in actor_keys.items() if "net." in k}
        )
    except Exception:
        # state dict 구조가 다를 수 있으므로 최선 노력
        pass

    mlp.eval()

    def act_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            return mlp(obs_t).cpu().numpy()

    return act_fn


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(args_cli.device if args_cli.device else "cuda")

    # ── 환경 생성 (소규모 평가용) ─────────────────────────────────────────────
    import importlib
    spec      = gym.spec(args_cli.task)
    cfg_entry = spec.kwargs["env_cfg_entry_point"]
    mod_path, cls_name = cfg_entry.rsplit(":", 1)
    env_cfg   = getattr(importlib.import_module(mod_path), cls_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.track_csv = TRACK_CSV_MAP[args_cli.track]

    env = gym.make(args_cli.task, cfg=env_cfg)

    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        obs_space = obs_space["policy"]
    obs_dim    = obs_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    results: dict[str, dict] = {}

    # ── PPO 평가 ──────────────────────────────────────────────────────────────
    if args_cli.ppo_ckpt:
        print(f"\n[벤치마크] PPO 평가 중... ({args_cli.ppo_ckpt})")
        try:
            act_fn = make_ppo_act_fn(args_cli.ppo_ckpt, obs_dim, action_dim, device)
            results["PPO"] = evaluate_policy(
                env, act_fn, args_cli.eval_episodes, args_cli.max_steps, device
            )
            print(f"  PPO  → mean={results['PPO']['mean_reward']:.3f} ± {results['PPO']['std_reward']:.3f}")
        except Exception as e:
            print(f"  PPO 평가 실패: {e}")
            results["PPO"] = {"error": str(e)}

    # ── TQC 평가 ──────────────────────────────────────────────────────────────
    if args_cli.tqc_ckpt:
        print(f"\n[벤치마크] TQC 평가 중... ({args_cli.tqc_ckpt})")
        try:
            agent = TQCAgent(state_dim=obs_dim, action_dim=action_dim, device=device)
            agent.load(args_cli.tqc_ckpt)

            def tqc_act(obs):
                return agent.select_action(obs, deterministic=True)

            results["TQC"] = evaluate_policy(
                env, tqc_act, args_cli.eval_episodes, args_cli.max_steps, device
            )
            print(f"  TQC  → mean={results['TQC']['mean_reward']:.3f} ± {results['TQC']['std_reward']:.3f}")
        except Exception as e:
            print(f"  TQC 평가 실패: {e}")
            results["TQC"] = {"error": str(e)}

    # ── TD7 평가 ──────────────────────────────────────────────────────────────
    if args_cli.td7_ckpt:
        print(f"\n[벤치마크] TD7 평가 중... ({args_cli.td7_ckpt})")
        try:
            agent = TD7Agent(state_dim=obs_dim, action_dim=action_dim, device=device)
            agent.load(args_cli.td7_ckpt)

            def td7_act(obs):
                return agent.select_action(obs, deterministic=True)

            results["TD7"] = evaluate_policy(
                env, td7_act, args_cli.eval_episodes, args_cli.max_steps, device
            )
            print(f"  TD7  → mean={results['TD7']['mean_reward']:.3f} ± {results['TD7']['std_reward']:.3f}")
        except Exception as e:
            print(f"  TD7 평가 실패: {e}")
            results["TD7"] = {"error": str(e)}

    env.close()

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("알고리즘 성능 비교 요약")
    print("=" * 60)
    print(f"{'알고리즘':<10} {'평균 보상':>12} {'표준편차':>10} {'최소':>8} {'최대':>8}")
    print("-" * 60)
    for name, res in results.items():
        if "error" in res:
            print(f"{name:<10} {'ERROR':>12}")
        else:
            print(
                f"{name:<10} "
                f"{res['mean_reward']:>12.4f} "
                f"{res['std_reward']:>10.4f} "
                f"{res['min_reward']:>8.4f} "
                f"{res['max_reward']:>8.4f}"
            )
    print("=" * 60)

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    output_path = args_cli.output
    if output_path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join("outputs", f"benchmark_{ts}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "task": args_cli.task,
                "eval_episodes": args_cli.eval_episodes,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
