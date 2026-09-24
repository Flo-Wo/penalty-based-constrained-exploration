"""
PGP experiment: maximum-entropy exploration on dm_control point_mass:easy
with an occupancy-measure KL-divergence constraint.

Problem solved:
    max  H(lambda^pi)
    s.t. D_KL(lambda^pi || lambda^ref) <= kappa

where lambda^ref is the occupancy measure of the SAC reference policy trained
by train_ref_policy.py.

The KL is estimated per-step as:
    g(s,a) = log d^pi(s) - log d^ref(s) + log pi(a|s) - log pi_ref(a|s) - kappa

and the constraint is E_{lambda^pi}[g(s,a)] <= 0.

Usage (after running train_ref_policy.py):
    python point_mass_main.py --kappa 1.0 --beta 10.0 --seed 0
    python point_mass_main.py --kappa 2.0 --beta 10.0 --seed 0
    python point_mass_main.py --kappa 5.0 --beta 10.0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import mediapy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC

from pgp_core import (
    BoxTanhGaussianPolicy,
    GMMStateDensity,
    StateBuffer,
    ValueNet,
    collect_rollout_gym,
    discount_cumsum,
    get_best_device,
    set_seed,
    shadow_reward_from_functionals,
    to_torch,
)
from point_mass_env import PointMassEasyGym
from metrics_utils import MetricsLogger, plot_training_metrics
from policy_utils import save_checkpoint


class SACLogProbWrapper:
    """
    Wraps a frozen SB3 SAC actor to provide the same log_prob(obs, act) interface
    as BoxTanhGaussianPolicy, so it can be used directly in the KL constraint.
    """

    def __init__(self, sac_model: SAC):
        self._actor = sac_model.policy.actor
        self._actor.eval()
        for p in self._actor.parameters():
            p.requires_grad_(False)

    def to(self, device: torch.device) -> "SACLogProbWrapper":
        self._actor.to(device)
        return self

    def log_prob(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        """
        Compute log pi_ref(a|s) for batched (obs, act) tensors.
        act_t must be in the environment action space (same as what env.step() receives).
        """
        mean, log_std, kwargs = self._actor.get_action_dist_params(obs_t)
        dist = self._actor.action_dist.proba_distribution(
            mean_actions=mean, log_std=log_std
        )
        return dist.log_prob(act_t)


@dataclass
class PointMassConfig:
    seed: int = 0
    device: str = get_best_device()

    max_ep_len: int = 1000

    vf_lr: float = 3e-4
    train_iters: int = 1000
    grad_clip: float = 1.0

    occ_steps_per_iter: int = 200
    occ_batch: int = 512
    state_buffer_size: int = 200_000

    beta: float = 10.0
    kappa: float = 1.0
    beta_growth: float = 1.0

    logd_clip_min: float = -50.0
    loglam_clip_min: float = -50.0

    critic_updates: int = 10
    critic_mb: int = 512

    checkpoint_dir: str = "./checkpoints_point_mass"
    checkpoint_freq: int = 100
    video_freq: int = 100
    save_final: bool = True

    ref_sac_path: str = "./checkpoints/ref_point_mass/ref_sac"
    ref_occ_path: str = "./checkpoints/ref_point_mass/ref_occ.pt"


def record_training_video(
    policy: "BoxTanhGaussianPolicy",
    device: torch.device,
    time_limit: float,
    save_path: str,
    seed: int = 0,
    fps: int = 30,
) -> None:
    """Roll out the current policy for one episode and save an MP4."""
    env = PointMassEasyGym(time_limit=time_limit, seed=seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        obs_t = to_torch(obs[None, :], device)
        with torch.no_grad():
            a, _ = policy.sample(obs_t)
        obs, _, terminated, truncated, _ = env.step(a.squeeze(0).cpu().numpy())
        done = terminated or truncated
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mediapy.write_video(save_path, frames, fps=fps)


def train(cfg: PointMassConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    run_name = f"kappa{cfg.kappa}_seed{cfg.seed}_beta{cfg.beta}"
    checkpoint_dir = os.path.join(cfg.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    metrics_logger = MetricsLogger(checkpoint_dir)

    print(f"Loading SAC reference policy from {cfg.ref_sac_path}.zip …")
    sac_model = SAC.load(cfg.ref_sac_path, device=device)
    policy_ref = SACLogProbWrapper(sac_model).to(device)

    print(f"Loading reference occupancy from {cfg.ref_occ_path} …")
    occ_ckpt = torch.load(cfg.ref_occ_path, map_location=device)
    obs_dim = occ_ckpt["obs_dim"]
    occ_K_ref = occ_ckpt["K"]

    occ_ref = GMMStateDensity(dim=obs_dim, K=occ_K_ref).to(device)
    occ_ref.load_state_dict(occ_ckpt["state_dict"])
    occ_ref.eval()
    for p in occ_ref.parameters():
        p.requires_grad_(False)

    print(f"Reference objects loaded  (obs_dim={obs_dim}, occ_K_ref={occ_K_ref}).")

    env = PointMassEasyGym(time_limit=cfg.time_limit, seed=cfg.seed)

    act_low = env.action_space.low
    act_high = env.action_space.high

    policy = BoxTanhGaussianPolicy(
        obs_dim=obs_dim, act_low=act_low, act_high=act_high
    ).to(device)
    vnet = ValueNet(obs_dim=obs_dim).to(device)
    occ = GMMStateDensity(dim=obs_dim, K=cfg.occ_K).to(device)

    pi_opt = optim.Adam(policy.parameters(), lr=cfg.pi_lr)
    vf_opt = optim.Adam(vnet.parameters(), lr=cfg.vf_lr)
    occ_opt = optim.Adam(occ.parameters(), lr=cfg.occ_lr)

    state_buf = StateBuffer(cfg.state_buffer_size, obs_dim)

    beta = cfg.beta

    def obj_entropy(lam, loglam):
        w = lam / (lam.mean() + 1e-12)
        return -(w * loglam).mean()

    def constr_kl_le_kappa(kl_per_step: torch.Tensor):
        """
        D_KL(lambda^pi || lambda^ref) - kappa <= 0.

        kl_per_step[t] = log d^pi(s_t) - log d^ref(s_t)
                        + log pi(a_t|s_t) - log pi_ref(a_t|s_t) - kappa
        Fully detached; autograd flows only through (lam, loglam).
        """
        g_det = kl_per_step.detach()

        def _c(lam, loglam):
            w = lam / (lam.mean() + 1e-12)
            return (w * g_det).mean()

        return _c

    for it in range(cfg.train_iters):

        rollout = collect_rollout_gym(env, policy, cfg, device)
        state_buf.add(rollout.obs)

        for _ in range(cfg.occ_steps_per_iter):
            s = to_torch(state_buf.sample(cfg.occ_batch), device)
            occ_loss = -occ.log_prob(s).mean()
            occ_opt.zero_grad(set_to_none=True)
            occ_loss.backward()
            occ_opt.step()

        obs_t = to_torch(rollout.obs, device)
        act_t = to_torch(rollout.act, device)

        with torch.no_grad():
            log_dHat = occ.log_prob(obs_t).clamp_min(cfg.logd_clip_min)
            log_dRef = occ_ref.log_prob(obs_t).clamp_min(cfg.logd_clip_min)
            log_piRef = policy_ref.log_prob(obs_t, act_t)

        log_pi = policy.log_prob(obs_t, act_t).detach()

        log_lamHat = (log_dHat + log_pi).clamp_min(cfg.loglam_clip_min)

        kl_per_step = (log_dHat + log_pi - log_dRef - log_piRef - cfg.kappa).detach()

        kl_raw = float((log_dHat + log_pi - log_dRef - log_piRef).mean().cpu())
        kl_violation = max(0.0, kl_raw - cfg.kappa)
        mean_reward = float(rollout.env_reward.mean())

        r_shadow_t, st = shadow_reward_from_functionals(
            log_lam=log_lamHat,
            obj_fn=obj_entropy,
            constr_fns=[constr_kl_le_kappa(kl_per_step)],
            beta=beta,
        )
        r_total = r_shadow_t.cpu().numpy().astype(np.float32)

        rtg = discount_cumsum(r_total, cfg.gamma, rollout.done)
        rtg_t = to_torch(rtg, device)

        T = rtg_t.shape[0]
        idx_all = torch.arange(T, device=device)

        for _ in range(cfg.critic_updates):
            mb = idx_all[torch.randint(0, T, (min(cfg.critic_mb, T),), device=device)]
            v_pred = vnet(obs_t[mb])
            vf_loss = 0.5 * ((v_pred - rtg_t[mb]) ** 2).mean()
            vf_opt.zero_grad(set_to_none=True)
            vf_loss.backward()
            nn.utils.clip_grad_norm_(vnet.parameters(), cfg.grad_clip)
            vf_opt.step()

        with torch.no_grad():
            adv = rtg_t - vnet(obs_t)

        logpi = policy.log_prob(obs_t, act_t)
        pi_loss = -(logpi * adv).mean()

        pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        pi_opt.step()

        beta *= cfg.beta_growth

        if cfg.checkpoint_freq > 0 and (it + 1) % cfg.checkpoint_freq == 0:
            save_checkpoint(
                policy,
                vnet,
                occ,
                pi_opt,
                vf_opt,
                occ_opt,
                it + 1,
                beta,
                cfg,
                checkpoint_dir,
            )

        if it < 10:
            vid_path = os.path.join(checkpoint_dir, "videos", f"iter_{it + 1:05d}.mp4")
            record_training_video(
                policy, device, cfg.time_limit, vid_path, seed=cfg.seed
            )
        if cfg.video_freq > 0 and (it + 1) % cfg.video_freq == 0:
            vid_path = os.path.join(checkpoint_dir, "videos", f"iter_{it + 1:05d}.mp4")
            record_training_video(
                policy, device, cfg.time_limit, vid_path, seed=cfg.seed
            )

        if it % 10 == 0:
            H_hat = float(-log_lamHat.mean().cpu())
            H_hat_old = float((-log_lamHat * torch.exp(log_lamHat)).mean().cpu())

            metrics_logger.log(
                iteration=it,
                entropy=H_hat,
                entropy_old=H_hat_old,
                mean_constraint=kl_raw,
                constraint_violation=kl_violation,
                penalty=st["pen"],
                occ_loss=occ_loss.item(),
                policy_loss=pi_loss.item(),
                critic_loss=vf_loss.item(),
                beta=beta,
                mean_reward=mean_reward,
            )
            metrics_logger.save()

            print(
                f"it={it:05d}  H={H_hat:+.3f}  KL={kl_raw:.3f} (budget={cfg.kappa:.2f})"
                f"  viol={kl_violation:.3f}  P={st['pen']:.3f}"
                f"  NLL={occ_loss.item():.3f}  pi_loss={pi_loss.item():.3f}"
            )

    if cfg.save_final:
        save_checkpoint(
            policy,
            vnet,
            occ,
            pi_opt,
            vf_opt,
            occ_opt,
            cfg.train_iters,
            beta,
            cfg,
            checkpoint_dir,
        )
        metrics_logger.save()
        print("Generating training curves …")
        plot_training_metrics(checkpoint_dir)
        vid_path = os.path.join(
            checkpoint_dir, "videos", f"iter_{cfg.train_iters:05d}_final.mp4"
        )
        record_training_video(policy, device, cfg.time_limit, vid_path, seed=cfg.seed)
        print(f"\nTraining complete!  Saved to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGP point_mass: KL-constrained max-entropy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--kappa", type=float, default=1.0, help="KL budget")
    parser.add_argument("--train_iters", type=int, default=1000)
    parser.add_argument("--steps_per_iter", type=int, default=4096)
    parser.add_argument("--beta_growth", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints_point_mass"
    )
    parser.add_argument(
        "--ref_sac_path", type=str, default="./checkpoints/ref_point_mass/ref_sac"
    )
    parser.add_argument(
        "--ref_occ_path", type=str, default="./checkpoints/ref_point_mass/ref_occ.pt"
    )
    parser.add_argument("--time_limit", type=float, default=20.0)
    parser.add_argument(
        "--video_freq",
        type=int,
        default=100,
        help="Save a training video every N iterations (0 = disabled)",
    )
    args = parser.parse_args()

    cfg = PointMassConfig(
        seed=args.seed,
        beta=args.beta,
        kappa=args.kappa,
        train_iters=args.train_iters,
        steps_per_iter=args.steps_per_iter,
        beta_growth=args.beta_growth,
        checkpoint_dir=args.checkpoint_dir,
        video_freq=args.video_freq,
        ref_sac_path=args.ref_sac_path,
        ref_occ_path=args.ref_occ_path,
        time_limit=args.time_limit,
    )
    train(cfg)
