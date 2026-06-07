"""
src/utils.py — Shared utilities for Team MOGU Mario RL
"""

import os
import yaml
import json
import numpy as np


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config and return as flat dict for easy access."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_metrics(metrics: dict, path: str = "results/metrics.json"):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {path}")


def load_metrics(path: str = "results/metrics.json") -> dict:
    with open(path) as f:
        return json.load(f)


def setup_dirs(*dirs):
    """Create required directories."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_summary(metrics: dict):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  TEAM MOGU — Results Summary")
    print(f"{'='*50}")
    for model_name, m in metrics.items():
        print(f"\n  {model_name}:")
        print(f"    Mean Reward    : {m.get('mean_reward', 0):.1f} ± {m.get('std_reward', 0):.1f}")
        print(f"    Mean X-Position: {m.get('mean_x_position', 0):.1f}")
        print(f"    Completions    : {m.get('completions', 0)}/{m.get('n_episodes', 100)}")
        print(f"    Inference FPS  : {m.get('inference_fps', 0):.1f}")
    print(f"\n{'='*50}\n")
