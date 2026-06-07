"""
evaluate.py — Reproduces headline number from checkpoints/best.pt
Team MOGU — DL2026 Final Project

Usage:
    python evaluate.py                          # uses checkpoints/best.pt
    python evaluate.py --path checkpoints/best.pt --episodes 100
"""

import argparse
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def evaluate(model_path: str = "checkpoints/best.pt",
             model_type: str = "ppo",
             n_episodes: int = 100,
             world: int = 1,
             stage: int = 1,
             deterministic: bool = True):

    from src.wrappers import make_eval_env

    # Load model
    if model_type == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    else:
        from stable_baselines3 import DQN
        model = DQN.load(model_path)

    print(f"\n{'='*50}")
    print(f"  Team MOGU — PPO Evaluation")
    print(f"  Checkpoint : {model_path}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  World-Stage: {world}-{stage}")
    print(f"{'='*50}\n")

    env = make_eval_env(world=world, stage=stage, action_type="right_only")

    rewards, x_positions, completions, lengths, times = [], [], [], [], []
    total_steps = 0
    total_time  = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward, ep_steps, max_x, completed = 0.0, 0, 0, False
        ep_start = time.time()

        while not done:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            total_time += time.time() - t0

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            total_steps += 1
            max_x = max(max_x, info.get("x_pos", 0))
            if info.get("flag_get", False):
                completed = True
            done = terminated or truncated

        rewards.append(ep_reward)
        x_positions.append(max_x)
        completions.append(int(completed))
        lengths.append(ep_steps)
        times.append(time.time() - ep_start)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes} | "
                  f"Reward: {ep_reward:7.1f} | "
                  f"X-pos: {max_x:4d} | "
                  f"Done: {'✅' if completed else '❌'}")

    env.close()
    fps = total_steps / total_time if total_time > 0 else 0

    # ── Headline results ──────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  HEADLINE RESULT")
    print(f"{'='*50}")
    print(f"  Mean Reward     : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Median Reward   : {np.median(rewards):.1f}")
    print(f"  Mean X-Position : {np.mean(x_positions):.1f}")
    print(f"  Max X-Position  : {np.max(x_positions):.0f}")
    print(f"  Completions     : {sum(completions)}/{n_episodes} "
          f"({sum(completions)/n_episodes*100:.1f}%)")
    print(f"  Inference FPS   : {fps:.1f}")
    print(f"{'='*50}\n")

    metrics = {
        "model": model_type.upper(),
        "checkpoint": model_path,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_x_position": float(np.mean(x_positions)),
        "max_x_position": float(np.max(x_positions)),
        "completions": int(sum(completions)),
        "completion_rate": float(sum(completions) / n_episodes),
        "inference_fps": float(fps),
    }

    # Save to results/metrics.json
    os.makedirs("results", exist_ok=True)
    existing = {}
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            existing = json.load(f)
    existing[model_type.upper()] = metrics
    with open("results/metrics.json", "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Results saved to results/metrics.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",       type=str, default="checkpoints/best.pt")
    parser.add_argument("--model",      type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--episodes",   type=int, default=100)
    parser.add_argument("--world",      type=int, default=1)
    parser.add_argument("--stage",      type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.path,
        model_type=args.model,
        n_episodes=args.episodes,
        world=args.world,
        stage=args.stage,
        deterministic=not args.stochastic,
    )
