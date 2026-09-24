"""
Shared primitives for PGP (Policy Gradient with Penalty) experiments.

Contains all environment-agnostic models and utilities used by both
the CartPole position-constraint experiment (main.py) and the
point-mass KL-constraint experiment (point_mass_main.py).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_best_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        print("using CUDA")
        return "cuda"
    if torch.backends.mps.is_available():
        print("using MPS")
        return "mps"
    print("using CPU")
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def discount_cumsum(x: np.ndarray, gamma: float, dones: np.ndarray) -> np.ndarray:
    """Compute discounted returns-to-go with episode termination."""
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        if dones[t]:
            running = 0.0
        running = x[t] + gamma * running
        out[t] = running
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, act=nn.Tanh):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoxTanhGaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy for Box action spaces.
    Maps N(mu, std) through tanh then affinely to [act_low, act_high].
    """

    def __init__(
        self,
        obs_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        hidden=(256, 256),
        log_std_bounds=(-5.0, 2.0),
    ):
        super().__init__()
        act_low = np.asarray(act_low, dtype=np.float32).reshape(-1)
        act_high = np.asarray(act_high, dtype=np.float32).reshape(-1)
        assert act_low.shape == act_high.shape
        self.act_dim = act_low.size

        self.mu_net = MLP(obs_dim, list(hidden), self.act_dim, act=nn.ReLU)
        self.log_std_net = MLP(obs_dim, list(hidden), self.act_dim, act=nn.ReLU)
        self.log_std_min, self.log_std_max = log_std_bounds

        scale = 0.5 * (act_high - act_low)
        bias = 0.5 * (act_high + act_low)

        self.register_buffer("act_scale", torch.as_tensor(scale))
        self.register_buffer("act_bias", torch.as_tensor(bias))
        self.register_buffer(
            "log_abs_det_scale", torch.log(torch.clamp(self.act_scale, min=1e-8)).sum()
        )

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_net(obs)
        log_std = torch.clamp(self.log_std_net(obs), self.log_std_min, self.log_std_max)
        return mu, torch.exp(log_std)

    def _tanh_to_env(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return self.act_bias + self.act_scale * a_tanh

    def _env_to_tanh(self, a_env: torch.Tensor) -> torch.Tensor:
        return (a_env - self.act_bias) / torch.clamp(self.act_scale, min=1e-8)

    @staticmethod
    def _log_prob_tanh(
        dist: torch.distributions.Normal, z: torch.Tensor, a_tanh: torch.Tensor
    ) -> torch.Tensor:
        logp_z = dist.log_prob(z).sum(dim=-1)
        eps = 1e-6
        log_det = torch.log(1 - a_tanh.pow(2) + eps).sum(dim=-1)
        return logp_z - log_det

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a_tanh = torch.tanh(z)
        logp_tanh = self._log_prob_tanh(dist, z, a_tanh)
        a_env = self._tanh_to_env(a_tanh)
        logp_env = logp_tanh - self.log_abs_det_scale
        return a_env, logp_env

    def log_prob(self, obs: torch.Tensor, a_env: torch.Tensor) -> torch.Tensor:
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        a_tanh = self._env_to_tanh(a_env)
        z = self._atanh(a_tanh)
        logp_tanh = self._log_prob_tanh(dist, z, a_tanh)
        return logp_tanh - self.log_abs_det_scale


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        self.v = MLP(obs_dim, list(hidden), 1, act=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class GMMStateDensity(nn.Module):
    """Diagonal-covariance GMM for occupancy density estimation."""

    def __init__(self, dim: int, K: int = 16, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.dim = dim
        self.K = K
        self.log_std_min, self.log_std_max = log_std_bounds

        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(torch.randn(K, dim) * 0.1)
        self.log_stds = nn.Parameter(torch.zeros(K, dim))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_stds = torch.clamp(self.log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds)

        x_exp = x[:, None, :]
        means = self.means[None, :, :]
        stds = stds[None, :, :]

        norm_const = -0.5 * self.dim * math.log(2 * math.pi)
        log_det = -torch.log(stds).sum(dim=-1)
        quad = -0.5 * (((x_exp - means) / stds) ** 2).sum(dim=-1)
        comp_logp = norm_const + log_det + quad

        log_w = torch.log_softmax(self.logits, dim=0)[None, :]
        return torch.logsumexp(log_w + comp_logp, dim=1)


@dataclass
class Rollout:
    obs: np.ndarray
    act: np.ndarray
    done: np.ndarray
    u: np.ndarray
    g: np.ndarray
    env_reward: np.ndarray


class StateBuffer:
    def __init__(self, max_size: int, obs_dim: int):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.ptr = 0
        self.size = 0
        self.data = np.zeros((max_size, obs_dim), dtype=np.float32)

    def add(self, obs: np.ndarray) -> None:
        n = obs.shape[0]
        for i in range(n):
            self.data[self.ptr] = obs[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> np.ndarray:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.data[idx]


def _reset_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(rew), done, info
    obs, rew, done, info = out
    return obs, float(rew), bool(done), info


def collect_rollout_gym(env, policy, cfg, device: torch.device) -> Rollout:
    """Collect a fixed number of environment steps using the current policy."""
    obs_list, act_list, done_list, u_list, g_list, rew_list = [], [], [], [], [], []

    obs, info = _reset_env(env)
    ep_len = 0

    for _ in range(cfg.steps_per_iter):
        obs_t = to_torch(obs[None, :], device)

        with torch.no_grad():
            a_env, _ = policy.sample(obs_t)
        a = a_env.squeeze(0).cpu().numpy()

        next_obs, env_rew, done, info = _step_env(env, a)

        g = float(info.get("constraint", 0.0))
        u = g

        obs_list.append(obs.copy())
        act_list.append(a.copy())
        done_list.append(float(done))
        u_list.append(u)
        g_list.append(g)
        rew_list.append(float(env_rew))

        obs = next_obs
        ep_len += 1
        if done or (ep_len >= cfg.max_ep_len):
            obs, info = _reset_env(env)
            ep_len = 0

    return Rollout(
        obs=np.asarray(obs_list, dtype=np.float32),
        act=np.asarray(act_list, dtype=np.float32),
        done=np.asarray(done_list, dtype=np.float32),
        u=np.asarray(u_list, dtype=np.float32),
        g=np.asarray(g_list, dtype=np.float32),
        env_reward=np.asarray(rew_list, dtype=np.float32),
    )


def shadow_reward_from_functionals(
    log_lam: torch.Tensor,
    obj_fn,
    constr_fns,
    beta: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute pseudo-rewards via automatic differentiation of the penalized functional.

    log_lam   : [T] log lambda-hat(s,a) at sampled pairs
    obj_fn    : (lam, log_lam) -> scalar  to MAXIMIZE
    constr_fns: list of (lam, log_lam) -> scalar  constraints c_j <= 0
    beta      : quadratic penalty weight

    Returns (shadow_rewards [T], diagnostics dict).
    """
    ell = log_lam.detach().requires_grad_(True)
    lam = torch.exp(ell)

    J = obj_fn(lam, ell)
    pen = torch.zeros(1, device=log_lam.device)
    for c_fn in constr_fns:
        c = c_fn(lam, ell)
        pen = pen + torch.relu(c) ** 2

    L = -(J - beta * pen)
    (g_ell,) = torch.autograd.grad(L, ell)

    r = -g_ell / (lam.detach() + 1e-12)
    return r.detach(), dict(J=float(J.detach()), pen=float(pen.detach()))
