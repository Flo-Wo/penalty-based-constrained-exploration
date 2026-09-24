from copy import deepcopy
from typing import Type
import os
import time

import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm, trange

from agent import Agent
from general_utility import (
    FUNCTION_REFERENCES,
    POLICY_REFERENCES,
    AGENT_REFERENCES,
    OPTIMIZER_REFERENCES,
    LR_SCHEDULER_REFERENCES,
)


class Experiment:
    """
    Class used to run and save experiments on different agents and environments.
    """

    def __init__(
        self,
        name,
        agent: Type[Agent],
        agent_kwargs: dict,
        objective_func: Type[nn.Module],
        objective_func_kwargs: dict,
        constraint_func: Type[nn.Module],
        constraint_func_kwargs: dict,
        env_id: str,
        env_kwargs: dict,
        max_episode_length,
        policy: Type[nn.Module],
        policy_kwargs: dict,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        lr_scheduler: Type[torch.optim.lr_scheduler._LRScheduler] | None = None,
        lr_scheduler_kwargs: dict = {"init": {}, "step": {}},
        checkpoint_step=100,
        custom_path=None,
        episodes_per_epoch=1,
    ):
        self.name = name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = policy(**policy_kwargs).to(self.device)
        self.optimizer = optimizer(params=self.policy.parameters(), **optimizer_kwargs)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(
                optimizer=self.optimizer, **lr_scheduler_kwargs["init"]
            )
        else:
            self.lr_scheduler = None
        self.env_id = env_id
        self.env = gym.make(env_id, **env_kwargs)
        self.objective_func = objective_func(**objective_func_kwargs).to(self.device)
        self.constraint_func = constraint_func(**constraint_func_kwargs).to(self.device)
        self.agent = agent(
            env=self.env,
            policy=self.policy,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs["step"],
            objective_func=self.objective_func,
            constraint_func=self.constraint_func,
            device=self.device,
            **agent_kwargs,
        )
        self.max_episode_length = max_episode_length
        self.episodes_per_epoch = episodes_per_epoch
        self.checkpoint_step = checkpoint_step
        if custom_path is None:
            self.path = f"experiments/{agent.__name__}/{env_id}/"
        else:
            assert isinstance(custom_path, str)
            self.path = custom_path + "/"
        self.experiment_kwargs = {
            "env_id": env_id,
            "env_kwargs": env_kwargs,
            "agent": agent.__name__,
            "agent_kwargs": agent_kwargs,
            "policy": policy.__name__,
            "policy_kwargs": policy_kwargs,
            "optimizer": optimizer.__name__,
            "optimizer_kwargs": optimizer_kwargs,
            "objective_func": objective_func.__name__,
            "objective_func_kwargs": objective_func_kwargs,
            "constraint_func": constraint_func.__name__,
            "constraint_func_kwargs": constraint_func_kwargs,
            "max_episode_length": max_episode_length,
            "checkpoint_step": checkpoint_step,
            "custom_path": custom_path,
            "episodes_per_epoch": episodes_per_epoch,
        }
        if lr_scheduler:
            self.experiment_kwargs["lr_scheduler"] = (lr_scheduler.__name__,)
            self.experiment_kwargs["lr_scheduler_kwargs"] = (lr_scheduler_kwargs,)

        self.save_state = {"experiment_kwargs": self.experiment_kwargs.copy()}
        self.save_state["checkpoints"] = []

        self.epoch = 0
        self.save_checkpoint(write_to_file=False)

    def save_checkpoint(self, write_to_file=True, retries=3) -> None:
        checkpoint = {
            "policy": deepcopy(self.policy.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "epoch": self.epoch,
            "vars": self.agent.get_vars(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = deepcopy(self.lr_scheduler.state_dict())
        self.save_state["checkpoints"].append(checkpoint)

        if write_to_file:
            try:
                if not os.path.isdir(self.path):
                    os.makedirs(self.path)
                torch.save(self.save_state, self.path + self.name)
            except:
                print("save failed! trying again...")
                time.sleep(1)
                torch.save(self.save_state, self.path + self.name)

    def load_from_file(self):
        try:
            self.save_state = torch.load(
                self.path + self.name, weights_only=True, map_location=self.device
            )
            self.load_checkpoint()
            return True
        except FileNotFoundError:
            print(f"(error) file not found: {self.path + self.name}")
            return False

    def load_checkpoint(self, index=-1):
        checkpoint = self.save_state["checkpoints"][index]
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch = checkpoint["epoch"]
        self.agent.set_vars(checkpoint["vars"])

    def run(self, n_epochs):
        pbar = tqdm(range(n_epochs), leave=False)
        for i in pbar:

            obj, constr = self.agent.step(
                self.episodes_per_epoch, self.max_episode_length
            )
            self.epoch += 1
            if (self.epoch % self.checkpoint_step) == 0:
                self.save_checkpoint()
                pbar.set_postfix_str(f"Objective: {obj:.2f} | Constraint: {constr:.2f}")

    def run_until(self, n_epochs):
        self.run(n_epochs - self.epoch)

    def eval_checkpoint(self, index, n_trajectories, max_episode_length=None):
        if max_episode_length is None:
            max_episode_length = self.max_episode_length
        self.load_checkpoint(index)
        return self.agent.eval(n_trajectories, max_episode_length)

    def eval_run(self, n_trajectories, max_episode_length=None):
        """Returns list of (epoch, (obj, constr)) tuples for all checkpoints."""
        vals_tensor = torch.empty(len(self.save_state["checkpoints"]), 3)
        for i in trange(len(self.save_state["checkpoints"]), leave=False):
            out = self.eval_checkpoint(i, n_trajectories, max_episode_length)
            vals_tensor[i, :] = torch.cat((torch.tensor([self.epoch]), out), dim=0)

        return vals_tensor


def already_trained(file_path, n_epochs):
    try:
        save_state = torch.load(file_path, weights_only=True)
        return save_state["checkpoints"][-1]["epoch"] >= n_epochs
    except FileNotFoundError:
        return False


def load_experiment(filepath, device="cpu"):
    kwargs = torch.load(filepath, weights_only=True, map_location=torch.device(device))[
        "experiment_kwargs"
    ]
    kwargs["agent"] = AGENT_REFERENCES[kwargs["agent"]]
    kwargs["policy"] = POLICY_REFERENCES[kwargs["policy"]]
    kwargs["objective_func"] = FUNCTION_REFERENCES[kwargs["objective_func"]]
    kwargs["constraint_func"] = FUNCTION_REFERENCES[kwargs["constraint_func"]]
    kwargs["optimizer"] = OPTIMIZER_REFERENCES[kwargs["optimizer"]]
    if "lr_scheduler" in kwargs:
        kwargs["lr_scheduler"] = LR_SCHEDULER_REFERENCES[kwargs["lr_scheduler"]]
    kwargs["custom_path"] = os.path.dirname(filepath)
    experiment = Experiment(name=os.path.basename(filepath), **kwargs)
    experiment.load_from_file()
    return experiment
