import numpy as np

import gymnasium as gym
from gymnasium import spaces

from dm_control import suite


def flatten_obs(obs_dict) -> np.ndarray:
    """Flatten dm_control OrderedDict observations into a 1D float32 vector."""
    return np.concatenate([np.asarray(v).ravel() for v in obs_dict.values()]).astype(
        np.float32
    )


class CartpoleSwingupSliderConstraintGym(gym.Env):
    """
    dm_control cartpole swingup wrapped as a Gym/Gymnasium env, with ONE constraint:
      slider_pos in [low, high]

    Constraint value returned in info:
      g = max(low - x, x - high)
      - g <= 0 => safe (negative margin)
      - g > 0  => violation magnitude
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        time_limit: float = 10.0,
        slider_pos_limits: tuple[float, float] = (-2.0, 2.0),
        seed: int | None = None,
        render_mode: str | None = None,
        camera_id: int = 0,
        height: int = 480,
        width: int = 640,
    ):
        super().__init__()
        self.low, self.high = float(slider_pos_limits[0]), float(slider_pos_limits[1])
        assert self.low < self.high, "slider_pos_limits must satisfy low < high"

        self.render_mode = render_mode
        self._camera_id = camera_id
        self._height = height
        self._width = width
        self._time_limit = float(time_limit)

        task_kwargs = {"time_limit": self._time_limit}
        if seed is not None:
            task_kwargs["random"] = np.random.RandomState(seed)

        self._env = suite.load(
            domain_name="cartpole",
            task_name="swingup",
            task_kwargs=task_kwargs,
        )

        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=np.asarray(act_spec.minimum, dtype=np.float32),
            high=np.asarray(act_spec.maximum, dtype=np.float32),
            shape=act_spec.shape,
            dtype=np.float32,
        )

        ts = self._env.reset()
        obs = flatten_obs(ts.observation)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def _constraint_value(self) -> float:
        """Compute signed constraint value g from physics (not from obs)."""
        x = float(np.asarray(self._env.physics.cart_position()).reshape(-1)[0])
        g = max(self.low - x, x - self.high)
        return float(g)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._env = suite.load(
                domain_name="cartpole",
                task_name="swingup",
                task_kwargs={
                    "time_limit": self._time_limit,
                    "random": np.random.RandomState(seed),
                },
            )

        ts = self._env.reset()
        obs = flatten_obs(ts.observation)

        info = {
            "constraint": self._constraint_value(),
            "slider_pos_limits": (self.low, self.high),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        ts = self._env.step(action)

        obs = flatten_obs(ts.observation)
        reward = float(ts.reward) if ts.reward is not None else 0.0

        terminated = False
        truncated = bool(ts.last())

        g = self._constraint_value()
        info = {
            "constraint": g,
            "constraint_violation": max(g, 0.0),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        rgb = self._env.physics.render(
            height=self._height, width=self._width, camera_id=self._camera_id
        )
        if self.render_mode == "rgb_array":
            return rgb

        if self.render_mode == "human":
            return rgb

        raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def close(self):
        pass
