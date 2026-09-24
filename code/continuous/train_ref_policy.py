"""
Train a reference policy for the point_mass:easy experiment using SAC (Stable-Baselines3).

The reference policy learns to reach the fixed target at the origin
purely from the dm_control reward signal.  After training:
  - The SB3 model is saved as  <save_dir>/ref_sac.zip
  - A GMMStateDensity (occ_ref) fit to the reference state visitation is saved as
    <save_dir>/ref_occ.pt

These two files are the only inputs required by point_mass_main.py.

Usage:
    python train_ref_policy.py                           # defaults
    python train_ref_policy.py --total_timesteps 300000
    python train_ref_policy.py --save_dir ./checkpoints/ref_point_mass
"""

from __future__ import annotations

import argparse
import os

import mediapy
import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from pgp_core import GMMStateDensity, get_best_device, set_seed, to_torch
from point_mass_env import PointMassEasyGym


# -------------------------
# Reference occupancy fitting
# -------------------------


def collect_ref_states(
    model: SAC,
    env: PointMassEasyGym,
    n_episodes: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Roll out the trained SAC policy deterministically and collect states."""
    all_obs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            all_obs.append(obs.copy())
    return np.array(all_obs, dtype=np.float32)


def fit_occ_ref(
    states: np.ndarray,
    K: int = 16,
    train_steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> GMMStateDensity:
    obs_dim = states.shape[1]
    occ = GMMStateDensity(obs_dim, K=K).to(device)
    opt = optim.Adam(occ.parameters(), lr=lr)
    states_t = to_torch(states, device)

    for step in range(train_steps):
        idx = torch.randint(0, len(states_t), (batch_size,))
        loss = -occ.log_prob(states_t[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  occ_ref fit  step={step:04d}  NLL={loss.item():.4f}")

    print(f"  occ_ref fit  final NLL={loss.item():.4f}")
    return occ


# -------------------------
# Video recording
# -------------------------


def record_ref_videos(
    model: SAC,
    save_dir: str,
    time_limit: float = 20.0,
    n_episodes: int = 5,
    seed: int = 0,
    fps: int = 30,
) -> None:
    """Roll out the trained SAC policy and save one MP4 per episode."""
    video_dir = os.path.join(save_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for ep in range(n_episodes):
        env = PointMassEasyGym(
            time_limit=time_limit, seed=seed + ep, render_mode="rgb_array"
        )
        obs, _ = env.reset(seed=seed + ep)
        frames = []
        done = False
        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        out_path = os.path.join(video_dir, f"ref_ep{ep:02d}.mp4")
        mediapy.write_video(out_path, frames, fps=fps)
        print(f"  Saved video: {out_path}  ({len(frames)} frames)")


# -------------------------
# Main training script
# -------------------------


def train_ref(
    total_timesteps: int = 200_000,
    save_dir: str = "./checkpoints/ref_point_mass",
    seed: int = 0,
    time_limit: float = 20.0,
    occ_K: int = 16,
    occ_episodes: int = 5000,
    device_str: str = "cpu",
) -> None:
    set_seed(seed)
    device = torch.device(device_str)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Train SAC
    # ------------------------------------------------------------------
    env = PointMassEasyGym(time_limit=time_limit, seed=seed)
    eval_env = PointMassEasyGym(time_limit=time_limit, seed=seed + 1000)

    print(f"Training SAC reference policy for {total_timesteps:,} timesteps …")
    print(
        f"  obs_dim={env.observation_space.shape[0]}  act_dim={env.action_space.shape[0]}"
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100_000,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy=-env.action_space.shape[0],  # standard SAC heuristic: -act_dim
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=10,
        verbose=1,
        deterministic=True,
    )

    print("START model.learn()")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("FINISHED model.learn()")

    # ------------------------------------------------------------------
    # Save SB3 model
    # ------------------------------------------------------------------
    sac_path = os.path.join(save_dir, "ref_sac")
    model.save(sac_path)
    print(f"Saved SAC model to {sac_path}.zip")

    # ------------------------------------------------------------------
    # Fit reference occupancy model
    # ------------------------------------------------------------------
    print(f"\nFitting reference occupancy model (K={occ_K}) …")
    ref_states = collect_ref_states(model, eval_env, n_episodes=occ_episodes, seed=seed)
    print(
        f"  Collected {len(ref_states)} reference states from {occ_episodes} episodes."
    )

    occ_ref = fit_occ_ref(ref_states, K=occ_K, device=device)

    occ_path = os.path.join(save_dir, "ref_occ.pt")
    torch.save(
        {
            "state_dict": occ_ref.state_dict(),
            "obs_dim": ref_states.shape[1],
            "K": occ_K,
        },
        occ_path,
    )
    print(f"Saved reference occupancy to {occ_path}")

    # ------------------------------------------------------------------
    # Record reference policy videos
    # ------------------------------------------------------------------
    print(f"\nRecording reference policy videos …")
    record_ref_videos(model, save_dir, time_limit=time_limit, n_episodes=5, seed=seed)

    print(f"\nReference training complete.  Files:")
    print(f"  {sac_path}.zip             ← SB3 SAC model")
    print(f"  {occ_path}                 ← GMM occupancy for log d^ref(s)")
    print(f"  {os.path.join(save_dir, 'videos')}  ← reference policy rollout videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC reference policy for point_mass:easy"
    )
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/ref_point_mass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit", type=float, default=20.0)
    parser.add_argument("--occ_K", type=int, default=16)
    parser.add_argument("--occ_episodes", type=int, default=5000)
    parser.add_argument("--device", type=str, default=get_best_device())
    args = parser.parse_args()

    train_ref(
        total_timesteps=args.total_timesteps,
        save_dir=args.save_dir,
        seed=args.seed,
        time_limit=args.time_limit,
        occ_K=args.occ_K,
        occ_episodes=args.occ_episodes,
        device_str=args.device,
    )
