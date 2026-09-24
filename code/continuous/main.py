from __future__ import annotations

import argparse
import os
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pendulum import CartpoleSwingupSliderConstraintGym
from debug_utils import visualize_policy_debug, plot_policy_behavior, render_episode
from metrics_utils import MetricsLogger, plot_training_metrics
from policy_utils import save_checkpoint
from pgp_core import (
    set_seed,
    get_best_device,
    to_torch,
    discount_cumsum,
    BoxTanhGaussianPolicy,
    ValueNet,
    GMMStateDensity,
    Rollout,
    StateBuffer,
    _reset_env,
    _step_env,
    collect_rollout_gym,
    shadow_reward_from_functionals,
)


@dataclass
class Config:
    seed: int = 0
    device: str = get_best_device()

    steps_per_iter: int = 4096
    max_ep_len: int = 1000

    gamma: float = 0.99
    pi_lr: float = 3e-4
    vf_lr: float = 3e-4
    train_iters: int = 2000
    grad_clip: float = 1.0

    occ_K: int = 16
    occ_lr: float = 1e-3
    occ_steps_per_iter: int = 200
    occ_batch: int = 512
    state_buffer_size: int = 200000

    beta: float = 10.0
    cost_budget: float = 0.0
    beta_growth: float = 1.00

    logd_clip_min: float = -50.0
    loglam_clip_min: float = -50.0

    critic_updates: int = 10
    critic_mb: int = 512

    checkpoint_dir: str = "./checkpoints"
    checkpoint_freq: int = 100
    save_final: bool = True


def train(cfg: Config, slider_limit: float = 1.5) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    _slider_limit = slider_limit
    slider_pos_limit = ((-1) * _slider_limit, _slider_limit)

    run_name = f"slider_pos_limit_{_slider_limit}_seed{cfg.seed}_beta{cfg.beta}"
    checkpoint_dir = os.path.join(cfg.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print(f"Saved config to {config_path}")

    metrics_logger = MetricsLogger(checkpoint_dir)

    env = CartpoleSwingupSliderConstraintGym(
        time_limit=10.0,
        slider_pos_limits=slider_pos_limit,
        seed=cfg.seed,
        render_mode=None,
    )

    obs0, info0 = env.reset()
    obs_dim = int(obs0.shape[0])

    act_low = env.action_space.low
    act_high = env.action_space.high
    policy = BoxTanhGaussianPolicy(obs_dim, act_low, act_high).to(device)

    vnet = ValueNet(obs_dim).to(device)
    occ = GMMStateDensity(obs_dim, K=cfg.occ_K).to(device)

    pi_opt = optim.Adam(policy.parameters(), lr=cfg.pi_lr)
    vf_opt = optim.Adam(vnet.parameters(), lr=cfg.vf_lr)
    occ_opt = optim.Adam(occ.parameters(), lr=cfg.occ_lr)

    state_buf = StateBuffer(cfg.state_buffer_size, obs_dim)

    beta = cfg.beta

    def obj_entropy(lam, loglam):
        w = lam / (lam.mean() + 1e-12)
        return -(w * loglam).mean()

    def constr_Eg_le_0(g_t: torch.Tensor):
        g_det = g_t.detach()

        def _c(lam, loglam):
            w = lam / (lam.mean() + 1e-12)
            return (w * g_det).mean()

        return _c

    for it in range(cfg.train_iters):

        rollout = collect_rollout_gym(env, policy, cfg, device)

        if it < 10 or it % 100 == 0:
            visualize_policy_debug(
                policy,
                rollout.obs,
                it,
                device,
                log_file=os.path.join(checkpoint_dir, "policy_debug.log"),
            )

        if it < 10 or it % 100 == 0:
            plot_policy_behavior(
                policy,
                rollout,
                it,
                device,
                save_dir=os.path.join(checkpoint_dir, "policy_plots"),
                slider_pos_limits=slider_pos_limit,
            )

        if it in [0, 1, 5, 10, 50, 100] or it % 100 == 0:
            env_render_kwargs = {
                "time_limit": 10.0,
                "slider_pos_limits": slider_pos_limit,
                "seed": it + 1000,
            }
            render_episode(
                policy,
                CartpoleSwingupSliderConstraintGym,
                env_render_kwargs,
                it,
                device,
                _reset_env,
                _step_env,
                save_dir=os.path.join(checkpoint_dir, "renders"),
                save_video=True,
                save_frames=False,
            )

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
        log_pi = policy.log_prob(obs_t, act_t).detach()
        log_lamHat = (log_dHat + log_pi).clamp_min(cfg.loglam_clip_min)

        Ju_hat = float(discount_cumsum(rollout.u, cfg.gamma, rollout.done).mean())
        v_hat = max(0.0, Ju_hat - cfg.cost_budget)
        mean_reward = float(rollout.env_reward.mean())

        g_t = to_torch(rollout.u, device)
        r_shadow_t, st = shadow_reward_from_functionals(
            log_lam=log_lamHat,
            obj_fn=obj_entropy,
            constr_fns=[constr_Eg_le_0(g_t)],
            beta=beta,
        )
        if it < 10:
            print(r_shadow_t)
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

        if it % 10 == 0:
            H_hat = float((-log_lamHat * torch.exp(log_lamHat)).mean().cpu())

            metrics_logger.log(
                iteration=it,
                entropy=float(H_hat),
                mean_constraint=Ju_hat,
                constraint_violation=v_hat,
                penalty=st["pen"],
                occ_loss=occ_loss.item(),
                policy_loss=pi_loss.item(),
                critic_loss=vf_loss.item(),
                beta=beta,
                mean_reward=mean_reward,
            )

            metrics_logger.save()

            print(
                f"it={it:05d} H={H_hat:+.3f} E[g]={Ju_hat:+.3f} (E[g]_+)={v_hat:+.3f} "
                f"P={st['pen']:+.3f} NLL_lamHat={occ_loss.item():+.3f} "
                f"pi_RL_loss_baseline={pi_loss.item():+.3f} critic_loss={vf_loss.item():+.3f}"
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
        print("Generating training curves...")
        plot_training_metrics(checkpoint_dir)
        print(f"\nTraining complete! Final checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CartPole with constraints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--slider_limit", type=float, default=1.5)
    parser.add_argument("--steps_per_iter", type=int, default=4096)
    parser.add_argument("--train_iters", type=int, default=2000)
    parser.add_argument("--beta_growth", type=float, default=1.0)
    parser.add_argument("--cost_budget", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        steps_per_iter=args.steps_per_iter,
        train_iters=args.train_iters,
        beta=args.beta,
        beta_growth=args.beta_growth,
        cost_budget=args.cost_budget,
        checkpoint_dir=args.checkpoint_dir,
    )
    train(cfg, slider_limit=args.slider_limit)
