"""
Minimal Gymnasium wrapper for dm_control point_mass:easy.

The task: drive a 2D point mass to a fixed target at the origin using 2D forces.
Target is fixed at (0, 0) in the 'easy' variant — no goal conditioning needed.

Observation (4D, native dm_control):
  [pos_x, pos_y, vel_x, vel_y]
  where pos is normalized by workspace half-size (0.29 m) to approx [-1, 1].

Action (2D): force in x and y, from the env action spec (nominally [-1, 1]^2).

This wrapper exists solely to provide a Gymnasium-compatible API.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite

_WORKSPACE_HALF = 0.29


def _flatten_obs(obs_dict) -> np.ndarray:
    """Flatten dm_control OrderedDict observation into a 1D float32 vector."""
    return np.concatenate([np.asarray(v).ravel() for v in obs_dict.values()]).astype(
        np.float32
    )


class PointMassEasyGym(gym.Env):
    """
    dm_control point_mass:easy wrapped as a Gym/Gymnasium environment.

    No constraint logic — the KL constraint is handled externally
    in the PGP training loop (point_mass_main.py).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        time_limit: float = 20.0,
        seed: int | None = None,
        render_mode: str | None = None,
        height: int = 480,
        width: int = 480,
    ):
        super().__init__()
        self._time_limit = float(time_limit)
        self.render_mode = render_mode
        self._height = height
        self._width = width

        self._env = self._make_env(seed)

        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=np.asarray(act_spec.minimum, dtype=np.float32),
            high=np.asarray(act_spec.maximum, dtype=np.float32),
            shape=act_spec.shape,
            dtype=np.float32,
        )

        ts = self._env.reset()
        obs0 = _flatten_obs(ts.observation)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32
        )

    def _make_env(self, seed: int | None):
        task_kwargs = {"time_limit": self._time_limit}
        if seed is not None:
            task_kwargs["random"] = np.random.RandomState(seed)
        return suite.load(
            domain_name="point_mass",
            task_name="easy",
            task_kwargs=task_kwargs,
        )

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._env = self._make_env(seed)
        ts = self._env.reset()
        return _flatten_obs(ts.observation), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        ts = self._env.step(action)
        obs = _flatten_obs(ts.observation)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = False
        truncated = bool(ts.last())
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self._height, width=self._width, camera_id=0
            )
        return None

    def close(self):
        pass

    def get_position(self) -> np.ndarray:
        """Return current (x, y) mass position in metres."""
        return np.asarray(self._env.physics.data.qpos[:2], dtype=np.float32)

    def get_position_normalized(self) -> np.ndarray:
        """Return current (x, y) mass position normalized by workspace half-size."""
        return self.get_position() / _WORKSPACE_HALF


if __name__ == "__main__":
    env = PointMassEasyGym(time_limit=20.0, seed=0)
    obs, _ = env.reset()
    print(f"obs shape : {obs.shape}")
    print(f"obs       : {obs}")
    print(f"act space : {env.action_space}")

    total_reward = 0.0
    for t in range(100):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        total_reward += r
        if terminated or truncated:
            break
    print(f"random policy reward over {t+1} steps: {total_reward:.4f}")
