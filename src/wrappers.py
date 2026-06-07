"""
wrappers.py — Environment preprocessing for Super Mario Bros
Handles: grayscale, frame skip, frame stack, resize, reward shaping
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque


class SkipFrame(gym.Wrapper):
    """Repeat the same action for `skip` frames and sum rewards."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB frames to grayscale."""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """Resize frames to a fixed shape for the CNN input."""
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)


class FrameStack(gym.ObservationWrapper):
    """Stack `n` consecutive frames to give the agent a sense of motion."""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_stack, *obs_shape),
            dtype=np.uint8
        )

    def observation(self, observation):
        self.frames.append(observation)
        return np.array(self.frames)

    def reset(self, **kwargs):
        # nes-py's JoypadSpace doesn't support seed/options kwargs
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return np.array(self.frames), info


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1]."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0


class ShapedRewardWrapper(gym.Wrapper):
    """
    Reward shaping to speed up learning significantly.

    Instead of sparse rewards, we give:
    - Continuous reward for moving right (progress)
    - Penalty for moving left or standing still
    - Small time penalty to encourage speed
    - Large bonus for completing the level
    - Clear penalty for dying
    - Small bonus for coins and score increases
    """
    def __init__(self, env):
        super().__init__(env)
        self._current_x = 0
        self._current_score = 0
        self._current_life = 2

    def reset(self, **kwargs):
        self._current_x = 0
        self._current_score = 0
        self._current_life = 2
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_pos    = info.get("x_pos", 0)
        score    = info.get("score", 0)
        life     = info.get("life", 2)
        flag_get = info.get("flag_get", False)

        shaped = 0.0

        # Progress: reward moving right, penalise moving left
        x_delta = x_pos - self._current_x
        shaped += x_delta * 0.5

        # Score delta (coins, enemies defeated)
        score_delta = score - self._current_score
        shaped += score_delta * 0.025

        # Time penalty — encourages finishing quickly
        shaped -= 0.05

        # Level completion bonus
        if flag_get:
            shaped += 500.0

        # Death penalty
        if life < self._current_life:
            shaped -= 50.0

        self._current_x = x_pos
        self._current_score = score
        self._current_life = life

        return obs, shaped, terminated, truncated, info


def _build_base_env(world, stage, action_type):
    """Build raw JoypadSpace env with nes-py compatibility patches."""
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

    action_map = {
        "right_only": RIGHT_ONLY,
        "simple":     SIMPLE_MOVEMENT,
        "complex":    COMPLEX_MOVEMENT,
    }

    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v0",
        apply_api_compatibility=True,
        render_mode="rgb_array",
    )
    env = JoypadSpace(env, action_map[action_type])

    # Patch 1: nes-py doesn't accept seed/options in reset()
    _orig_reset = env.reset
    def _patched_reset(**kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        return _orig_reset(**kwargs)
    env.reset = _patched_reset

    # Patch 2: convert old gym spaces to gymnasium spaces for SB3 2.x
    import gymnasium
    env.action_space = gymnasium.spaces.Discrete(env.action_space.n)
    env.observation_space = gymnasium.spaces.Box(
        low=env.observation_space.low,
        high=env.observation_space.high,
        shape=env.observation_space.shape,
        dtype=env.observation_space.dtype,
    )

    return env


def make_mario_env(
    world=1, stage=1,
    action_type="simple",       # upgraded from right_only
    skip=4,
    resize_shape=(84, 84),
    n_stack=4,
    shaped_rewards=True,        # reward shaping on by default
    normalize=True,
):
    """
    Build the full preprocessed Mario environment.

    action_type='simple' gives 7 actions including jump+stop+run
    which lets the agent learn to dodge enemies properly.
    """
    env = _build_base_env(world, stage, action_type)

    # Reward shaping before frame skip so shaping sees every frame's info
    if shaped_rewards:
        env = ShapedRewardWrapper(env)

    env = SkipFrame(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=resize_shape)
    env = FrameStack(env, n_stack=n_stack)

    if normalize:
        env = NormalizeObservation(env)

    return env


def make_eval_env(world=1, stage=1, action_type="simple"):
    """
    Evaluation env — shaped rewards off so we measure true game score.
    Same visual preprocessing as training env.
    """
    env = _build_base_env(world, stage, action_type)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, n_stack=4)
    env = NormalizeObservation(env)
    return env
