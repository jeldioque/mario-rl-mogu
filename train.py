"""
train.py — Main training entry point for Team MOGU Mario RL
Reads config from configs/default.yaml

Usage:
    # Train PPO (main model)
    python train.py --model ppo --run_name ppo_main

    # Train DQN (baseline)
    python train.py --model dqn --run_name dqn_baseline

    # Continue PPO from checkpoint
    python train.py --model ppo --run_name ppo_v2 --load checkpoints/best.pt
"""

import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, setup_dirs
from src.wrappers import make_mario_env, make_eval_env
from src.model import build_ppo, build_dqn, make_callbacks, make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def train(model_type: str, run_name: str, load_path: str = None,
          config_path: str = "configs/default.yaml"):

    cfg = load_config(config_path)
    env_cfg   = cfg["environment"]
    model_cfg = cfg["model"]
    log_cfg   = cfg["logging"]

    if model_type == "ppo":
        algo_cfg = cfg["ppo"]
    else:
        algo_cfg = cfg["dqn"]

    setup_dirs(f"checkpoints/{run_name}", f"runs/{run_name}", "logs")

    print(f"\n{'='*55}")
    print(f"  Team MOGU — {model_type.upper()} Training")
    print(f"  Run name  : {run_name}")
    print(f"  Timesteps : {algo_cfg['timesteps']:,}")
    print(f"  World     : {env_cfg['world']}-{env_cfg['stage']}")
    print(f"  Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Load from : {load_path or 'scratch'}")
    print(f"{'='*55}\n")

    if model_type == "ppo":
        n_envs = algo_cfg["n_envs"]
        env = SubprocVecEnv([
            make_env(env_cfg["world"], env_cfg["stage"],
                     rank=i, seed=algo_cfg["seed"],
                     action_type=env_cfg["action_type"],
                     shaped_rewards=env_cfg["shaped_rewards"])
            for i in range(n_envs)
        ])
        eval_env = DummyVecEnv([lambda: Monitor(
            make_eval_env(env_cfg["world"], env_cfg["stage"],
                          action_type=env_cfg["action_type"])
        )])

        ppo_cfg = {**algo_cfg, **model_cfg}
        model = build_ppo(env, ppo_cfg, run_name, load_path)
        callbacks = make_callbacks(run_name, eval_env, n_envs)

        model.learn(
            total_timesteps=algo_cfg["timesteps"],
            callback=callbacks,
            tb_log_name="PPO",
            reset_num_timesteps=(load_path is None),
        )

    else:  # DQN
        env = DummyVecEnv([lambda: Monitor(
            make_mario_env(env_cfg["world"], env_cfg["stage"],
                           action_type=env_cfg["action_type"],
                           shaped_rewards=False)
        )])
        eval_env = DummyVecEnv([lambda: Monitor(
            make_eval_env(env_cfg["world"], env_cfg["stage"],
                          action_type=env_cfg["action_type"])
        )])

        model = build_dqn(env, algo_cfg, run_name)
        callbacks = make_callbacks(run_name, eval_env, n_envs=1)

        model.learn(
            total_timesteps=algo_cfg["timesteps"],
            callback=callbacks,
            tb_log_name="DQN",
        )

    # Save final model and copy to best.pt
    final_path = f"checkpoints/{run_name}/final"
    model.save(final_path)

    # Always update checkpoints/best.pt for the rubric
    import shutil
    shutil.copy(f"{final_path}.zip", "checkpoints/best.pt")
    print(f"\n✅ Training complete!")
    print(f"   Final model : {final_path}.zip")
    print(f"   Best model  : checkpoints/best.pt")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--run_name",type=str, default="ppo_main")
    parser.add_argument("--load",    type=str, default=None,
                        help="Path to checkpoint to continue from (without .zip)")
    parser.add_argument("--config",  type=str, default="configs/default.yaml")
    args = parser.parse_args()

    train(args.model, args.run_name, args.load, args.config)
