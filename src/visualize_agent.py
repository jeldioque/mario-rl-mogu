"""
visualize_agent.py — Watch your PPO agent play Super Mario Bros
Team MOGU

Usage:
  # Watch live (requires display)
  python visualize_agent.py --model checkpoints/ppo_v2/best_model

  # Record to video (headless-friendly)
  python visualize_agent.py --model checkpoints/ppo_v2/best_model --record --out replay.mp4

  # Multiple episodes, specific stage
  python visualize_agent.py --model checkpoints/ppo_v2/best_model --episodes 5 --world 1 --stage 2

  # DQN baseline (right_only action space)
  python visualize_agent.py --model checkpoints/dqn_baseline/best_model --algo dqn
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

# ── Bring models/ onto path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from models.wrappers import make_eval_env


# ── Overlay helpers ─────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (80,  220,  80)
RED    = (80,   80, 220)
YELLOW = (40,  220, 220)
WHITE  = (240, 240, 240)
DARK   = ( 20,  20,  20)
CYAN   = (220, 200,  60)


def put_text(frame, text, pos, color=WHITE, scale=0.55, thickness=1):
    x, y = pos
    # Shadow
    cv2.putText(frame, text, (x+1, y+1), FONT_SMALL, scale, DARK, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),     FONT_SMALL, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, stats: dict, ep: int, total_eps: int) -> np.ndarray:
    """
    Draws a semi-transparent HUD panel over the rendered frame.

    stats keys expected:
      step, reward, x_pos, score, life, flag_get, deaths,
      best_x, total_reward
    """
    h, w = frame.shape[:2]

    # Panel background (top strip)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Episode counter
    ep_str = f"EP {ep}/{total_eps}"
    put_text(frame, ep_str, (8, 18), CYAN, scale=0.5)

    # Step + reward
    put_text(frame, f"Step : {stats['step']:>5}", (8,  36), WHITE, scale=0.45)
    put_text(frame, f"Rew  : {stats['reward']:>+7.1f}", (8, 52), YELLOW, scale=0.45)

    # x_pos progress bar
    bar_x0, bar_y = 90, 22
    bar_w, bar_h  = w - bar_x0 - 8, 10
    max_x         = 3266          # approximate level width for 1-1
    fill          = int(bar_w * min(stats['x_pos'] / max_x, 1.0))
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill,  bar_y + bar_h), GREEN, -1)
    put_text(frame, f"x:{stats['x_pos']:>4}", (bar_x0 + bar_w + 4, bar_y + 9), GREEN, scale=0.38)

    # Score / Lives / Deaths
    put_text(frame, f"Score:{stats['score']:>6}", (90, 40), WHITE, scale=0.42)
    life_col = GREEN if stats['life'] >= 2 else RED
    put_text(frame, f"Lives:{stats['life']}", (90, 54), life_col, scale=0.42)
    put_text(frame, f"Deaths:{stats['deaths']}", (w - 90, 18), RED, scale=0.42)
    put_text(frame, f"Best x:{stats['best_x']:>4}", (w - 90, 34), CYAN, scale=0.42)

    # Flag flash
    if stats.get("flag_get"):
        cv2.rectangle(frame, (0, 58), (w, h), (0, 180, 0), 6)
        msg = "LEVEL COMPLETE!"
        tw  = cv2.getTextSize(msg, FONT, 1.2, 2)[0][0]
        cv2.putText(frame, msg, ((w - tw)//2, h//2), FONT, 1.2, (40, 255, 40), 2, cv2.LINE_AA)

    return frame


# ── Single episode runner ────────────────────────────────────────────────────

def run_episode(model, env, render: bool, scale: float, delay: float):
    """
    Run one episode. Returns dict with summary stats and a list of frames
    (only populated when render=False i.e. recording mode).
    """
    obs, info = env.reset()
    done   = False
    step   = 0
    total_reward = 0.0
    best_x = 0
    deaths = 0
    flag   = False
    prev_life = info.get("life", 2)
    frames = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        step         += 1
        total_reward += reward
        x_pos  = info.get("x_pos",   0)
        score  = info.get("score",   0)
        life   = info.get("life",    2)
        flag   = info.get("flag_get", False)
        best_x = max(best_x, x_pos)

        if life < prev_life:
            deaths += 1
        prev_life = life

        stats = dict(
            step=step, reward=total_reward, x_pos=x_pos,
            score=score, life=life, flag_get=flag,
            deaths=deaths, best_x=best_x, total_reward=total_reward,
        )

        # Grab the raw RGB frame from the underlying env
        raw = env.env.render()          # Monitor wraps the wrappers chain
        if raw is None:
            raw = np.zeros((240, 256, 3), dtype=np.uint8)

        raw = raw.copy()
        frame_hud = draw_hud(raw, stats, ep=0, total_eps=0)  # ep filled by caller

        if render:
            disp = cv2.resize(
                frame_hud,
                (int(frame_hud.shape[1] * scale), int(frame_hud.shape[0] * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("MOGU — PPO Mario Agent", disp)
            key = cv2.waitKey(max(1, int(delay * 1000)))
            if key == ord('q'):
                return None, None   # caller treats None as quit signal
        else:
            frames.append(frame_hud)

        time.sleep(delay if render else 0)

    summary = dict(
        steps=step, total_reward=total_reward,
        best_x=best_x, flag=flag, deaths=deaths,
    )
    return summary, frames


# ── Video writer helper ──────────────────────────────────────────────────────

def save_video(all_frames: list, out_path: str, fps: int = 30):
    if not all_frames:
        print("  ⚠️  No frames captured — skipping video save.")
        return
    h, w = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in all_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"\n  🎬 Video saved → {out_path}  ({len(all_frames)} frames @ {fps} fps)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watch or record your trained Mario agent."
    )
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to model zip (without .zip)")
    parser.add_argument("--algo",     type=str, default="ppo", choices=["ppo", "dqn"],
                        help="Algorithm used to train the model (default: ppo)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run (default: 3)")
    parser.add_argument("--world",    type=int, default=1)
    parser.add_argument("--stage",    type=int, default=1)
    parser.add_argument("--scale",    type=float, default=3.0,
                        help="Display scale factor (default: 3× for crisp pixels)")
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="Target FPS for live display / video (default: 30)")
    parser.add_argument("--record",   action="store_true",
                        help="Record gameplay to video instead of showing live")
    parser.add_argument("--out",      type=str, default="mario_replay.mp4",
                        help="Output video path (default: mario_replay.mp4)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path_zip = model_path + ".zip"
    else:
        model_path_zip = model_path

    if not os.path.exists(model_path_zip):
        print(f"❌ Model not found: {model_path_zip}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  MOGU — Agent Visualizer")
    print(f"  Model     : {model_path}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  World-Stage: {args.world}-{args.stage}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Mode      : {'🎬 Record → ' + args.out if args.record else '🖥️  Live display'}")
    print(f"{'='*55}\n")

    # Action type: DQN baseline uses right_only, PPO uses simple
    action_type = "right_only" if args.algo == "dqn" else "simple"

    AlgoClass = DQN if args.algo == "dqn" else PPO
    device    = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(f"  Loading model on {device}...")
    model = AlgoClass.load(model_path, device=device)
    print("  ✅ Model loaded.\n")

    delay  = 1.0 / args.fps
    render = not args.record

    all_results = []
    all_frames  = []

    for ep in range(1, args.episodes + 1):
        print(f"  ── Episode {ep}/{args.episodes} ──")

        # Fresh env each episode so wrappers reset cleanly
        env = make_eval_env(args.world, args.stage, action_type=action_type)
        env = Monitor(env)

        summary, frames = run_episode(model, env, render=render, scale=args.scale, delay=delay)
        env.close()

        if summary is None:
            print("  [q pressed — stopping early]")
            break

        # Annotate frames with episode number (post-hoc)
        if frames:
            for f in frames:
                put_text(f, f"EP {ep}/{args.episodes}", (8, 18), CYAN, scale=0.5)
            all_frames.extend(frames)

        all_results.append(summary)
        print(f"     Steps : {summary['steps']}")
        print(f"     Reward: {summary['total_reward']:.1f}")
        print(f"     Best x: {summary['best_x']}")
        print(f"     Deaths: {summary['deaths']}")
        print(f"     Flag  : {'✅ YES' if summary['flag'] else '❌ no'}\n")

    if render:
        cv2.destroyAllWindows()

    # ── Record mode: save video ────────────────────────────────────────────
    if args.record and all_frames:
        save_video(all_frames, args.out, fps=int(args.fps))

    # ── Summary table ─────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*55}")
        print(f"  SUMMARY  ({len(all_results)} episodes)")
        print(f"{'='*55}")
        rewards = [r["total_reward"] for r in all_results]
        x_poss  = [r["best_x"]       for r in all_results]
        flags   = sum(r["flag"]       for r in all_results)
        deaths  = sum(r["deaths"]     for r in all_results)
        print(f"  Mean reward : {np.mean(rewards):.1f}  (std {np.std(rewards):.1f})")
        print(f"  Mean best-x : {np.mean(x_poss):.0f}")
        print(f"  Completions : {flags}/{len(all_results)}")
        print(f"  Total deaths: {deaths}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()