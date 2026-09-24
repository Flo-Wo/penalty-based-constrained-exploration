import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt

FUNCTION_REFERENCES = {}
POLICY_REFERENCES = {}
AGENT_REFERENCES = {}
OPTIMIZER_REFERENCES = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
}
LR_SCHEDULER_REFERENCES = {}


def register_function(func):
    FUNCTION_REFERENCES[func.__name__] = func
    return func


def register_policy(pol):
    POLICY_REFERENCES[pol.__name__] = pol
    return pol


def register_agent(agent):
    AGENT_REFERENCES[agent.__name__] = agent
    return agent


@register_function
class Entropy(nn.Module):
    def forward(self, lambda_tensor):
        log_lambda = torch.log(lambda_tensor + 1e-18)
        return -torch.sum(lambda_tensor * log_lambda)


@register_function
class NegativeEntropy(nn.Module):
    def forward(self, lambda_tensor):
        log_lambda = torch.log(lambda_tensor + 1e-18)
        return torch.sum(lambda_tensor * log_lambda)


@register_function
class Linear(nn.Module):
    def __init__(self, map):
        super().__init__()
        self.map = map

    def forward(self, lambda_tensor):
        return torch.sum(lambda_tensor * self.map)


@register_function
class NegativeLinear(nn.Module):
    def __init__(self, map):
        super().__init__()
        self.map = map

    def forward(self, lambda_tensor):
        return (-1) * torch.sum(lambda_tensor * self.map)


@register_function
class StateEntropy(nn.Module):
    def forward(self, lambda_tensor):
        lambda_tensor = lambda_tensor.sum(1)
        return Entropy(lambda_tensor)


@register_function
class NullFunction(nn.Module):
    def forward(self, lambda_tensor):
        return 0 * torch.sum(lambda_tensor)


@register_function
class NegativeDistance(nn.Module):
    def __init__(self, lambda_target: torch.Tensor, p: int = 2, shift: float = 0.0):
        super().__init__()
        self.lambda_target = torch.flatten(lambda_target)
        self.p = p
        self.shift = shift

    def forward(self, lambda_tensor):
        return (
            -torch.dist(torch.flatten(lambda_tensor), self.lambda_target, self.p)
            + self.shift
        )


@register_function
class Distance(nn.Module):
    """Encodes a constraint ||lambda - lambda_exp|| <= shift"""

    def __init__(self, lambda_target: torch.Tensor, p: int = 2, shift: float = 0.0):
        super().__init__()
        self.lambda_target = torch.flatten(lambda_target)
        self.p = p
        self.shift = shift

    def forward(self, lambda_tensor):
        return (
            torch.dist(torch.flatten(lambda_tensor), self.lambda_target, self.p)
            - self.shift
        )


@register_function
class KLDivergence(nn.Module):
    """Encodes constraint KL(lambda || lambda_target) <= shift"""

    def __init__(self, target: torch.Tensor, shift: float = 0):
        super().__init__()
        self.target = torch.flatten(target)
        self.shift = shift

    def forward(self, lambda_tensor):
        lambda_flat = torch.flatten(lambda_tensor)
        return (
            torch.sum(
                lambda_flat
                * (torch.log(lambda_flat + 1e-18) - torch.log(self.target + 1e-18))
            )
            - self.shift
        )


@register_function
class NegativeKLDivergence(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = torch.flatten(target)

    def forward(self, lambda_tensor):
        lambda_flat = torch.flatten(lambda_tensor)
        return (-1) * torch.sum(
            lambda_flat
            * (torch.log(lambda_flat + 1e-18) - torch.log(self.target + 1e-18))
        )


# step 1) compute the estimate of lambda
def compute_occupancy_measure(trajectories, state_space, action_space, discount_factor):
    lambda_tensor = torch.zeros(state_space, action_space)
    for trajectory in trajectories:
        states = trajectory["states"]
        actions = trajectory["actions"]
        for t, (s, a) in enumerate(zip(states, actions)):
            lambda_tensor[(int(s), int(a))] += 1.0 * (discount_factor**t)
    return lambda_tensor / len(trajectories)


def shadow_rewards(functional_in_lambda, lambdaHat: torch.Tensor):
    lambdaHat.requires_grad_(True)
    f_lambda = functional_in_lambda(lambdaHat)
    nabla_lambda_f_obj = torch.autograd.grad(
        f_lambda,
        lambdaHat,
        create_graph=True,
    )[0].detach()
    lambdaHat.requires_grad_(False)
    return f_lambda, nabla_lambda_f_obj


def discounted_cumsum(rewards, discount_factor):
    weights = discount_factor ** torch.arange(len(rewards)).to(rewards.device)
    x = rewards * weights
    return torch.flip(torch.cumsum(torch.flip(x, (0,)), 0), (0,))


def policy_grad_loss(
    trajectories: dict, rewards: torch.tensor, discount_factor: float
) -> torch.Tensor:
    loss = 0
    for trajectory in trajectories:
        rewards_inds = zip(trajectory["states"], trajectory["actions"])
        rewards_vec = torch.tensor(
            [
                rewards[(int(state_ind), int(act_ind))]
                for state_ind, act_ind in rewards_inds
            ]
        ).to(rewards.device)
        weights = discounted_cumsum(rewards_vec, discount_factor)
        loss += -torch.sum(weights * trajectory["logps"])
    return loss / len(trajectories)


def importance_sampling(
    trajectory: dict, target_policy: nn.Module, obs_dim: int
) -> torch.Tensor:
    with torch.no_grad():
        obs = F.one_hot(trajectory["states"].to(torch.long), obs_dim).to(torch.float)
        sum = 0
        for i, r in enumerate(obs):
            a = int(trajectory["actions"][i])
            sum += torch.log(target_policy(r)[a])
        return torch.exp(sum - torch.sum(trajectory["logps"]))


def plot_experiment(experiment, n_trajectories, title):
    vals = experiment.eval_run(n_trajectories, experiment.max_episode_length)
    epochs = [val[0] for val in vals]
    obj_vals = [val[1][0] for val in vals]
    constr_vals = [val[1][1] for val in vals]

    fig, ax = plt.subplots(2, 1)
    fig.supxlabel("epochs")
    for a in ax:
        a.grid()

    ax[0].plot(epochs, obj_vals)
    ax[0].set_ylabel("objective")

    ax[1].plot(epochs, constr_vals)
    ax[1].set_ylabel("constraint")
    ax[1].axhline(0, color="red")

    fig.suptitle(title)

    return fig, ax
