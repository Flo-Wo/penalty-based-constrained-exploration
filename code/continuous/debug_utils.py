"""
Debugging and visualization utilities for policy training.
"""

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, policy visualization will be limited")

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print(
        "Warning: imageio not available, video rendering will be disabled. Install with: pip install imageio"
    )


def to_torch(x: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.as_tensor(x, device=device, dtype=dtype)


def visualize_policy_debug(policy, obs_sample, it, device, n_samples=5, log_file=None):
    """Quick policy debugging visualization at beginning of training.

    Args:
        log_file: Optional file path to save logs. If provided, logs are written to file.
    """
    import os

    log_messages = []
    log_messages.append(f"\n=== Policy Debug @ iteration {it} ===")

    with torch.no_grad():
        obs_t = to_torch(obs_sample[:n_samples], device)

        actions, logprobs = policy.sample(obs_t)

        mu, std = policy.forward(obs_t)

        log_messages.append(f"Sample observations (first {n_samples}):")
        log_messages.append(f"  obs shape: {obs_sample.shape}")
        log_messages.append(f"  obs[0]: {obs_sample[0]}")

        log_messages.append(f"\nPolicy outputs:")
        log_messages.append(f"  mu (mean): {mu[0].cpu().numpy()}")
        log_messages.append(f"  std: {std[0].cpu().numpy()}")
        log_messages.append(f"  sampled action: {actions[0].cpu().numpy()}")
        log_messages.append(f"  log_prob: {logprobs[0].cpu().numpy():.4f}")

        log_messages.append(f"\nAction statistics across {n_samples} samples:")
        log_messages.append(f"  mean: {actions.mean(dim=0).cpu().numpy()}")
        log_messages.append(f"  std: {actions.std(dim=0).cpu().numpy()}")
        log_messages.append(f"  min: {actions.min(dim=0)[0].cpu().numpy()}")
        log_messages.append(f"  max: {actions.max(dim=0)[0].cpu().numpy()}")

    for msg in log_messages:
        print(msg)

    if log_file is not None:
        os.makedirs(
            os.path.dirname(log_file) if os.path.dirname(log_file) else ".",
            exist_ok=True,
        )
        with open(log_file, "a") as f:
            for msg in log_messages:
                f.write(msg + "\n")


def plot_policy_behavior(
    policy, rollout, it, device, save_dir="./policy_plots", slider_pos_limits=None
):
    """Create plots of policy behavior and observations for deeper debugging.

    Assumes observation structure from dm_control cartpole:
    obs = [cart_position, cart_velocity, pole_angle, pole_velocity]

    Args:
        slider_pos_limits: Optional tuple (low, high) for constraint visualization
    """
    if not HAS_MATPLOTLIB:
        return

    import os

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        obs_t = to_torch(rollout.obs[:500], device)
        mu, std = policy.forward(obs_t)
        actions, logprobs = policy.sample(obs_t)

        mu_np = mu.cpu().numpy()
        std_np = std.cpu().numpy()
        actions_np = actions.cpu().numpy()
        logprobs_np = logprobs.cpu().numpy()
        obs_np = rollout.obs[:500]

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        timesteps = np.arange(len(obs_np))

        axes[0, 0].plot(timesteps, mu_np, alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title(f"Policy Mean - Iter {it}", fontsize=11, fontweight="bold")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Action Mean")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(timesteps, std_np, alpha=0.7, linewidth=1.5)
        axes[0, 1].set_title(f"Policy Std - Iter {it}", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Action Std")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(
            timesteps, actions_np, alpha=0.7, linewidth=1.5, label="Sampled"
        )
        axes[1, 0].plot(
            timesteps,
            rollout.act[:500],
            alpha=0.5,
            linestyle="--",
            linewidth=1.5,
            label="Actual",
        )
        axes[1, 0].set_title(f"Actions - Iter {it}", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("Action Value")
        axes[1, 0].legend(loc="best")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(timesteps, logprobs_np, alpha=0.7, linewidth=1.5)
        axes[1, 1].set_title(
            f"Log Probabilities - Iter {it}", fontsize=11, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Timestep")
        axes[1, 1].set_ylabel("Log Prob")
        axes[1, 1].grid(True, alpha=0.3)

        if obs_np.shape[1] >= 1:
            cart_pos = obs_np[:, 0]
            axes[2, 0].plot(
                timesteps,
                cart_pos,
                alpha=0.7,
                linewidth=1.5,
                color="green",
                label="Cart Position",
            )

            if slider_pos_limits is not None:
                low, high = slider_pos_limits
                axes[2, 0].axhline(
                    y=low,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Lower Limit ({low})",
                )
                axes[2, 0].axhline(
                    y=high,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Upper Limit ({high})",
                )
                axes[2, 0].fill_between(
                    timesteps, low, high, alpha=0.1, color="green", label="Safe Region"
                )

            axes[2, 0].set_title(
                "Cart Position - Iter {it}", fontsize=11, fontweight="bold"
            )
            axes[2, 0].set_xlabel("Timestep")
            axes[2, 0].set_ylabel("Position (m)")
            axes[2, 0].legend(loc="best", fontsize=9)
            axes[2, 0].grid(True, alpha=0.3)

        if obs_np.shape[1] >= 3:
            sin_angle = obs_np[:, 1]
            cos_angle = obs_np[:, 2]
            angle = np.arctan2(sin_angle, cos_angle)
            axes[2, 1].plot(
                timesteps, np.degrees(angle), alpha=0.7, linewidth=1.5, color="orange"
            )
            axes[2, 1].set_title(
                "Pendulum Angle (sin/cos) - Iter {it}", fontsize=11, fontweight="bold"
            )
            axes[2, 1].set_xlabel("Timestep")
            axes[2, 1].set_ylabel("Angle (degrees)")
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].axhline(
                y=0, color="r", linestyle="--", alpha=0.3, label="Zero (downward)"
            )
            axes[2, 1].axhline(
                y=180, color="r", linestyle="--", alpha=0.3, label="180° (upright)"
            )
            axes[2, 1].axhline(y=-180, color="r", linestyle="--", alpha=0.3)
            axes[2, 1].legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/policy_iter_{it:04d}.png", dpi=100, bbox_inches="tight"
        )
        plt.close()
        print(f"  Saved policy plot to {save_dir}/policy_iter_{it:04d}.png")


def render_episode(
    policy,
    env_class,
    env_kwargs,
    it,
    device,
    _reset_env,
    _step_env,
    save_dir="./renders",
    max_steps=500,
    save_video=True,
    save_frames=False,
):
    """Render a full episode using current policy.

    Args:
        save_dir: Directory to save videos (can be checkpoint directory)
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    render_env = env_class(**env_kwargs, render_mode="rgb_array")

    obs, info = _reset_env(render_env)
    print("Reset in render episode")
    print(obs)
    frames = []
    rewards = []
    constraints = []
    actions_taken = []
    done = False
    step = 0

    print(f"\n=== Rendering Episode @ iteration {it} ===")

    q_list = []
    q_dot_list = []
    physics = render_env._env.physics

    with torch.no_grad():
        while not done and step < max_steps:
            frame = render_env.render()
            frames.append(frame)

            obs_t = to_torch(obs[None, :], device)
            a_env, _ = policy.sample(obs_t)
            a = a_env.squeeze(0).cpu().numpy()
            actions_taken.append(a.copy())

            obs, rew, done, info = _step_env(render_env, a)

            qpos = physics.data.qpos.copy()
            qvel = physics.data.qvel.copy()

            q_list.append(qpos)
            q_dot_list.append(qvel)

            rewards.append(rew)
            constraints.append(info.get("constraint", 0.0))
            step += 1

    total_reward = sum(rewards)
    max_constraint = max(constraints)
    mean_constraint = sum(constraints) / len(constraints)
    violations = sum(1 for c in constraints if c > 0)

    print(f"  Episode length: {step} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Mean constraint: {mean_constraint:.4f}")
    print(f"  Max constraint: {max_constraint:.4f}")
    print(f"  Constraint violations: {violations}/{step} steps")
    print(
        f"  Action range: [{min(min(a) for a in actions_taken):.3f}, {max(max(a) for a in actions_taken):.3f}]"
    )

    print(f"Number of of Frames: {len(frames)}")
    if save_video and HAS_IMAGEIO and len(frames) > 0:
        try:
            video_path = f"{save_dir}/episode_iter_{it:04d}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  Saved video to {video_path}")
        except (ValueError, RuntimeError) as e:
            print(
                f"  Warning: MP4 save failed ({str(e)[:50]}...), saving as GIF instead"
            )
            video_path = f"{save_dir}/episode_iter_{it:04d}.gif"
            imageio.mimsave(video_path, frames, fps=30, loop=0)
            print(f"  Saved video to {video_path}")
    elif save_video and not HAS_IMAGEIO:
        print("  Warning: imageio not available, cannot save video")

    if save_frames and HAS_MATPLOTLIB and len(frames) > 0:
        frame_dir = f"{save_dir}/frames_iter_{it:04d}"
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames[::10]):
            plt.imsave(f"{frame_dir}/frame_{i:04d}.png", frame)
        print(f"  Saved {len(frames[::10])} frames to {frame_dir}/")

    render_env.close()
    return {
        "total_reward": total_reward,
        "episode_length": step,
        "mean_constraint": mean_constraint,
        "max_constraint": max_constraint,
        "violation_rate": violations / step if step > 0 else 0,
        "q_pos": q_list,
        "q_vel": q_dot_list,
    }
