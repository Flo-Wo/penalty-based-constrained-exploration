from typing import Tuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions.categorical import Categorical

from tqdm import tqdm
import copy
from abc import ABC, abstractmethod

import general_utility as util


@util.register_policy
class LinearSoftmaxPolicy(nn.Module):
    def __init__(self, n: int, m: int) -> None:
        """n x m linear layer"""
        super().__init__()
        self.linear = nn.Linear(n, m, bias=True)

    def forward(self, x: torch.Tensor):
        if single_eval := x.dim() == 1:
            x = x[None, :]
        assert x.dim() == 2
        x = self.linear(x)
        if single_eval:
            return F.softmax(x, dim=1)[0]
        else:
            return F.softmax(x, dim=1)


class Agent(ABC):
    def __init__(
        self,
        env: gym.Env,
        discount_factor: float,
        policy: nn.Module,
        objective_func: nn.Module,
        constraint_func: nn.Module,
        optimizer: torch.optim.Optimizer,
        device,
        lr_scheduler: lr_scheduler._LRScheduler | None = None,
        lr_scheduler_kwargs: dict = {},
    ) -> None:
        self.env = env
        self.discount_factor = discount_factor
        self.objective_func = objective_func
        self.constraint_func = constraint_func

        self.n = env.observation_space.n
        self.m = env.action_space.n
        self.policy = policy

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.device = device

    def get_state_dist(self, obs: int):
        obs_tensor = F.one_hot(torch.tensor(obs, device=self.device), self.n).to(
            torch.float
        )
        return Categorical(self.policy(obs_tensor))

    def get_action(self, obs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ### input:
        - obs : observation
        ### output:
        - action : sampled from the policy distribution at obs
        - logp : the log probability of observing action
        """
        dist = self.get_state_dist(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_trajectory(self, length: int) -> dict:
        trajectory = {
            "states": torch.empty(length + 1, device=self.device),
            "actions": torch.empty(length, device=self.device),
            "logps": torch.empty(length, device=self.device),
        }

        obs, _ = self.env.reset()
        terminated, truncated = False, False
        iteration = 0

        while not (terminated or truncated) and iteration < length:
            trajectory["states"][iteration] = obs

            action, logp = self.get_action(obs)
            action = action.item()
            obs, _, terminated, truncated, _ = self.env.step(action)

            trajectory["actions"][iteration] = action
            trajectory["logps"][iteration] = logp
            iteration += 1

        # truncate trajectory length
        trajectory["states"] = trajectory["states"][:iteration]
        trajectory["actions"] = trajectory["actions"][:iteration]
        trajectory["logps"] = trajectory["logps"][:iteration]

        return trajectory

    def approx_occupancy_measure(self, n_episodes: int, length: int):
        trajectories = [self.get_trajectory(length) for _ in range(n_episodes)]
        lambdaHat = util.compute_occupancy_measure(
            trajectories, self.n, self.m, self.discount_factor
        ).to(self.device)

        return trajectories, lambdaHat

    def get_gradients(self) -> torch.Tensor:
        grads = [param.grad.detach().clone() for param in self.policy.parameters]
        return grads

    def set_gradients(self, grads: torch.Tensor) -> None:
        for grad, param in zip(grads, self.policy.parameters):
            param.grad = grad

    def step_from_trajectories(self, trajectories, lambdaHat) -> Tuple[float, float]:
        raise NotImplementedError

    def step(self, n_episodes: int, length: int) -> Tuple[float, float]:
        trajectories, lambdaHat = self.approx_occupancy_measure(n_episodes, length)

        return self.step_from_trajectories(trajectories, lambdaHat)

    def eval(self, n_episodes, length):
        self.policy.eval()
        with torch.no_grad():
            trajectories, lambdaHat = self.approx_occupancy_measure(n_episodes, length)
            obj = self.objective_func(lambdaHat)
            constr = self.constraint_func(lambdaHat)
            return torch.tensor((obj, constr))

    @abstractmethod
    def get_vars(self) -> dict:
        pass

    @abstractmethod
    def set_vars(self, vars):
        pass


@util.register_agent
class PDPG(Agent):
    def __init__(
        self,
        dual_lr: float,
        dual_bound: float = 1e7,
        dual_init: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dual_lr = dual_lr
        self.dual_bound = dual_bound
        self.dual_var = dual_init

    def step_from_trajectories(self, trajectories, lambdaHat) -> Tuple[float, float]:
        self.optimizer.zero_grad()

        obj, obj_rew = util.shadow_rewards(self.objective_func, lambdaHat)
        constr, constr_rew = util.shadow_rewards(self.constraint_func, lambdaHat)
        rewards = obj_rew + self.dual_var * constr_rew
        loss = util.policy_grad_loss(trajectories, rewards, self.discount_factor).to(
            self.device
        )
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step(self.lr_scheduler_kwargs["step"])

        # dual variable update
        with torch.no_grad():
            g = self.constraint_func(lambdaHat).detach().cpu()
            self.dual_var = np.clip(
                self.dual_var - self.dual_lr * g, 0, self.dual_bound
            )
        return obj, constr

    def get_vars(self):
        return {"dual_var": self.dual_var}

    def set_vars(self, vars):
        if "dual_var" in vars:
            self.dual_var = vars["dual_var"]


@util.register_agent
class VRPDPG(PDPG):
    """Variance-Reduced Primal-Dual Policy Gradient Algorithm (VR-PDPG)"""

    def __init__(
        self,
        alpha_t: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_t = alpha_t

        self.prev_policy = copy.deepcopy(self.policy)
        self.ref_lambda = None
        self.ref_trajectories = None
        self.ref_obj_reward = None
        self.ref_constr_reward = None

        self.lambda_t = None
        self.prev_obj_reward = None
        self.prev_constr_reward = None
        self.iteration = 0

    def step(self, n_episodes: int, length: int) -> Tuple[float, float]:
        """Execute one step of VR-PDPG algorithm"""

        if self.iteration == 0:
            self.ref_trajectories = [
                self.get_trajectory(length) for _ in range(n_episodes)
            ]

            self.ref_lambda = util.compute_occupancy_measure(
                self.ref_trajectories, self.n, self.m, self.discount_factor
            ).to(self.device)

            obj_val, self.ref_obj_reward = util.shadow_rewards(
                self.objective_func, self.ref_lambda
            )
            constr_val, self.ref_constr_reward = util.shadow_rewards(
                self.constraint_func, self.ref_lambda
            )

            self.prev_obj_reward = self.ref_obj_reward
            self.prev_constr_reward = self.ref_constr_reward

            rewards_0 = self.ref_obj_reward + self.dual_var * self.ref_constr_reward

            self.optimizer.zero_grad()
            loss = util.policy_grad_loss(
                self.ref_trajectories, rewards_0, self.discount_factor
            ).to(self.device)
            loss.backward()

            grad_norm = self._compute_grad_norm()
            if grad_norm > 0:
                self._scale_gradients(1.0 / grad_norm)

            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step(self.lr_scheduler_kwargs.get("step"))

            with torch.no_grad():
                g = self.constraint_func(self.ref_lambda).detach().cpu()
                self.dual_var = np.clip(
                    self.dual_var - self.dual_lr * g, 0, self.dual_bound
                )

            self.iteration += 1
            return obj_val.item(), constr_val.item()

        trajectories_t = [self.get_trajectory(length) for _ in range(n_episodes)]

        lambda_hat_t = util.compute_occupancy_measure(
            trajectories_t, self.n, self.m, self.discount_factor
        ).to(self.device)

        weighted_lambda = torch.zeros_like(lambda_hat_t)
        for traj in trajectories_t:
            w = util.importance_sampling(traj, self.prev_policy, self.n)
            traj_lambda = (
                util.compute_occupancy_measure(
                    [traj], self.n, self.m, self.discount_factor
                ).to(self.device)
                * n_episodes
            )
            weighted_lambda += w * traj_lambda
        weighted_lambda /= n_episodes

        if self.lambda_t is None:
            self.lambda_t = self.ref_lambda

        self.lambda_t = lambda_hat_t + (1 - self.alpha_t) * (
            self.lambda_t - weighted_lambda
        )

        obj_val, obj_reward_t = util.shadow_rewards(self.objective_func, self.lambda_t)
        constr_val, constr_reward_t = util.shadow_rewards(
            self.constraint_func, self.lambda_t
        )

        self.optimizer.zero_grad()
        rewards_current = self.prev_obj_reward + self.dual_var * self.prev_constr_reward
        loss_current = util.policy_grad_loss(
            trajectories_t, rewards_current, self.discount_factor
        ).to(self.device)
        loss_current.backward()
        grad_current = self._get_gradients()

        self.optimizer.zero_grad()
        weighted_loss = 0
        for traj in trajectories_t:
            w = util.importance_sampling(traj, self.prev_policy, self.n)
            rewards_prev = self.ref_obj_reward + self.dual_var * self.ref_constr_reward

            rewards_inds = zip(traj["states"], traj["actions"])
            rewards_vec = torch.tensor(
                [
                    rewards_prev[(int(state_ind), int(act_ind))]
                    for state_ind, act_ind in rewards_inds
                ]
            ).to(self.device)
            weights_disc = util.discounted_cumsum(rewards_vec, self.discount_factor)
            weighted_loss += -w * torch.sum(weights_disc * traj["logps"])
        weighted_loss /= n_episodes
        weighted_loss.backward()
        grad_weighted = self._get_gradients()

        if not hasattr(self, "prev_grad"):
            self.prev_grad = grad_current

        final_grads = []
        for g_curr, g_prev, g_weighted in zip(
            grad_current, self.prev_grad, grad_weighted
        ):
            g_vr = g_curr + (1 - self.alpha_t) * (g_prev - g_weighted)
            final_grads.append(g_vr)

        self.optimizer.zero_grad()
        self._set_gradients(final_grads)

        grad_norm = self._compute_grad_norm()
        if grad_norm > 0:
            self._scale_gradients(1.0 / grad_norm)

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step(self.lr_scheduler_kwargs.get("step"))

        with torch.no_grad():
            g = self.constraint_func(self.lambda_t).detach().cpu()
            self.dual_var = np.clip(
                self.dual_var - self.dual_lr * g, 0, self.dual_bound
            )

        self.ref_obj_reward = self.prev_obj_reward
        self.ref_constr_reward = self.prev_constr_reward
        self.prev_obj_reward = obj_reward_t
        self.prev_constr_reward = constr_reward_t
        self.prev_grad = final_grads
        self.prev_policy = copy.deepcopy(self.policy)
        self.iteration += 1

        return obj_val.item(), constr_val.item()

    def _get_gradients(self) -> list:
        """Get current gradients from policy parameters"""
        return [
            (
                param.grad.detach().clone()
                if param.grad is not None
                else torch.zeros_like(param)
            )
            for param in self.policy.parameters()
        ]

    def _set_gradients(self, grads: list) -> None:
        """Set gradients for policy parameters"""
        for param, grad in zip(self.policy.parameters(), grads):
            param.grad = grad.clone()

    def _compute_grad_norm(self) -> float:
        """Compute the norm of current gradients"""
        total_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def _scale_gradients(self, scale: float) -> None:
        """Scale all gradients by a factor"""
        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad.data *= scale

    def get_vars(self):
        vars = super().get_vars()
        vars.update(
            {
                "iteration": self.iteration,
                "lambda_t": self.lambda_t,
                "ref_lambda": self.ref_lambda,
                "prev_obj_reward": (
                    self.prev_obj_reward if hasattr(self, "prev_obj_reward") else None
                ),
                "prev_constr_reward": (
                    self.prev_constr_reward
                    if hasattr(self, "prev_constr_reward")
                    else None
                ),
            }
        )
        return vars

    def set_vars(self, vars):
        super().set_vars(vars)
        if "iteration" in vars:
            self.iteration = vars["iteration"]
        if "lambda_t" in vars:
            self.lambda_t = vars["lambda_t"]
        if "ref_lambda" in vars:
            self.ref_lambda = vars["ref_lambda"]
        if "prev_obj_reward" in vars:
            self.prev_obj_reward = vars["prev_obj_reward"]
        if "prev_constr_reward" in vars:
            self.prev_constr_reward = vars["prev_constr_reward"]


class SmoothQuadraticPlusPenalty(nn.Module):
    def __init__(self, constraint_func):
        super().__init__()
        self.constraint_func = constraint_func

    def forward(self, lambda_tensor):
        constr = self.constraint_func(lambda_tensor)
        return F.relu(constr) ** 2


@util.register_agent
class PolicyGradientPenaltyMethod(Agent):
    """
    Solve a Constrained General Utility RL problem of the form
    ```latex
    max_{\theta \in \Theta} F_1(\theta)
    s.t. F_2(\theta) \leq 0
    ```
    via a penalty approach.
    """

    def __init__(
        self,
        penalty_param: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.penalty_param = penalty_param

        self.previous_policy = copy.deepcopy(self.policy.state_dict())
        self.penalty_func = SmoothQuadraticPlusPenalty(self.constraint_func)
        self.pol_dist = 0.0
        self.loss = 0.0

    def step(self, n_episodes: int, length: int) -> Tuple[float, float]:
        self.optimizer.zero_grad()
        trajectories = [self.get_trajectory(length) for _ in range(n_episodes)]

        lambdaHat = util.compute_occupancy_measure(
            trajectories, self.n, self.m, self.discount_factor
        ).to(self.device)

        obj, obj_rew = util.shadow_rewards(self.objective_func, lambdaHat)
        constr = self.constraint_func(lambdaHat)
        penalty, penalty_rew = util.shadow_rewards(self.penalty_func, lambdaHat)

        rewards = obj_rew - self.penalty_param / 2.0 * penalty_rew
        loss = util.policy_grad_loss(trajectories, rewards, self.discount_factor)

        pol_dist = 0.0
        for key, val in self.policy.state_dict().items():
            pol_dist += torch.dist(val, self.previous_policy[key]) ** 2

        loss.backward()

        self.set_vars({"pol_dist": pol_dist, "loss": loss.detach()})

        self.optimizer.step()
        self.previous_policy = copy.deepcopy(self.policy.state_dict())
        if self.lr_scheduler:
            self.lr_scheduler.step(self.lr_scheduler_kwargs)
        return obj, constr

    def get_vars(self):
        return {"pol_dist": self.pol_dist, "loss": self.loss}

    def set_vars(self, vars):
        self.pol_dist = vars["pol_dist"]
        self.loss = vars["loss"]


@util.register_agent
class ProximalPointPolicyGradientPenaltyMethod(Agent):
    """
    Solve a Constrained General Utility RL problem of the form
    ```latex
    max_{\theta \in \Theta} F_1(\theta)
    s.t. F_2(\theta) \leq 0
    ```
    via a proximal point penalty approach, i.e. for every inner_T
    we solve a quadratically regularized penalty problem.
    """

    def __init__(
        self,
        penalty_param: float,
        inner_T: int,
        reg_param: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_T = inner_T
        self.inner_n = 0
        self.penalty_param = penalty_param
        self.reg_param = reg_param

        self.previous_policy = copy.deepcopy(self.policy.state_dict())
        self.penalty_func = SmoothQuadraticPlusPenalty(self.constraint_func)

        self.pol_dist = 0.0
        self.loss = 0.0

    def step(self, n_episodes: int, length: int) -> Tuple[float, float]:
        self.optimizer.zero_grad()
        trajectories = [self.get_trajectory(length) for _ in range(n_episodes)]

        self.inner_n = (self.inner_n + 1) % self.inner_T
        if self.inner_n == 0:
            self.previous_policy = copy.deepcopy(self.policy.state_dict())

        # approximate occupancy measure
        lambdaHat = util.compute_occupancy_measure(
            trajectories, self.n, self.m, self.discount_factor
        ).to(self.device)

        obj, obj_rew = util.shadow_rewards(self.objective_func, lambdaHat)
        constr = self.constraint_func(lambdaHat)
        penalty, penalty_rew = util.shadow_rewards(self.penalty_func, lambdaHat)

        rewards = obj_rew - self.penalty_param / 2.0 * penalty_rew
        loss = util.policy_grad_loss(trajectories, rewards, self.discount_factor)

        # regularization + log distance to previous policy
        pol_dist = 0.0
        for key, val in self.policy.state_dict().items():
            pol_dist += torch.dist(val, self.previous_policy[key]) ** 2

        loss -= self.reg_param / 2.0 * pol_dist
        loss.backward()

        self.set_vars({"pol_dist": pol_dist, "loss": loss.detach()})

        # policy update
        self.optimizer.step()
        self.previous_policy = copy.deepcopy(self.policy.state_dict())
        if self.lr_scheduler:
            self.lr_scheduler.step(self.lr_scheduler_kwargs)
        return obj, constr

    def get_vars(self):
        return {"pol_dist": self.pol_dist, "loss": self.loss}

    def set_vars(self, vars):
        self.pol_dist = vars["pol_dist"]
        self.loss = vars["loss"]
