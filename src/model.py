"""
src/model.py — Model architectures for Team MOGU Mario RL
Contains: DeepMarioCNN (3-layer CNN), ShallowCNN (baseline), PPO and DQN agent builders
"""

import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed


# ─────────────────────────────────────────────
# CNN Architectures
# ─────────────────────────────────────────────

class DeepMarioCNN(BaseFeaturesExtractor):
    """
    3-layer CNN — main PPO model.
    Input : (4, 84, 84) stacked grayscale frames
    Output: 512-dim feature vector

    Layer 1: detects large structures (platforms, pipes, ground)
    Layer 2: detects mid-scale patterns (enemies, Mario, blocks)
    Layer 3: detects fine features (edges, gaps, coins)
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ShallowCNN(BaseFeaturesExtractor):
    """
    2-layer CNN — DQN baseline.
    Deliberately simpler than DeepMarioCNN for comparison.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

class MarioStatsCallback(BaseCallback):
    """Tracks Mario-specific metrics: x-position, completions, deaths."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_x_positions = []
        self.episode_lengths = []
        self.completions = 0
        self.deaths = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_x_positions.append(info.get("x_pos", 0))
                self.episode_lengths.append(info["episode"]["l"])
                if info.get("flag_get", False):
                    self.completions += 1
                if info.get("life", 3) < 3 or info.get("status", "") == "dead":
                    self.deaths += 1
            if len(self.episode_x_positions) >= 10:
                self.logger.record("mario/mean_x_position", np.mean(self.episode_x_positions[-10:]))
                self.logger.record("mario/mean_ep_length", np.mean(self.episode_lengths[-10:]))
                self.logger.record("mario/total_completions", self.completions)
                self.logger.record("mario/total_deaths", self.deaths)
        return True


class EntropyAnnealCallback(BaseCallback):
    """Linearly decay entropy coefficient to encourage exploitation over time."""
    def __init__(self, ent_coef_start=0.02, ent_coef_end=0.001, verbose=0):
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = self.ent_coef_start + progress * (self.ent_coef_end - self.ent_coef_start)
        self.logger.record("train/ent_coef_annealed", self.model.ent_coef)
        return True


# ─────────────────────────────────────────────
# Environment factories
# ─────────────────────────────────────────────

def make_env(world=1, stage=1, rank=0, seed=42, action_type="right_only", shaped_rewards=True):
    def _init():
        from src.wrappers import make_mario_env
        set_random_seed(seed + rank)
        env = make_mario_env(world=world, stage=stage,
                             action_type=action_type,
                             shaped_rewards=shaped_rewards)
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────
# PPO builder
# ─────────────────────────────────────────────

def build_ppo(env, cfg: dict, run_name: str, load_path: str = None):
    """Build or load a PPO model from config dict."""
    policy_kwargs = dict(
        features_extractor_class=DeepMarioCNN,
        features_extractor_kwargs=dict(features_dim=cfg["features_dim"]),
        net_arch=dict(pi=cfg["net_arch_pi"], vf=cfg["net_arch_vf"]),
        activation_fn=nn.ReLU,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if load_path and os.path.exists(f"{load_path}.zip"):
        print(f"  Loading from: {load_path}")
        model = PPO.load(load_path, env=env, device=device,
                         learning_rate=cfg["learning_rate"],
                         n_steps=cfg["n_steps"],
                         batch_size=cfg["batch_size"],
                         tensorboard_log=f"runs/{run_name}")
        model.set_env(env)
    else:
        model = PPO(
            policy="CnnPolicy", env=env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            target_kl=cfg["target_kl"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"runs/{run_name}",
            verbose=1, seed=cfg["seed"], device=device,
        )
    return model


# ─────────────────────────────────────────────
# DQN builder
# ─────────────────────────────────────────────

def build_dqn(env, cfg: dict, run_name: str):
    """Build a DQN model from config dict."""
    policy_kwargs = dict(
        features_extractor_class=ShallowCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return DQN(
        policy="CnnPolicy", env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        target_update_interval=cfg["target_update_interval"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run_name}",
        verbose=1, seed=cfg["seed"], device=device,
    )


def make_callbacks(run_name, eval_env, n_envs, eval_freq=10_000, ckpt_freq=100_000):
    return CallbackList([
        EvalCallback(eval_env,
                     best_model_save_path=f"checkpoints/{run_name}",
                     log_path="logs",
                     eval_freq=max(eval_freq // n_envs, 1),
                     n_eval_episodes=20,
                     deterministic=True, render=False),
        CheckpointCallback(save_freq=max(ckpt_freq // n_envs, 1),
                           save_path=f"checkpoints/{run_name}",
                           name_prefix="checkpoint"),
        MarioStatsCallback(),
        EntropyAnnealCallback(),
    ])
