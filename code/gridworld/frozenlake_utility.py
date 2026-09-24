import torch
import numpy as np
from agent import Agent
from experiments import Experiment
import general_utility as util
import matplotlib.pyplot as plt
from PIL import Image
import os


def remove_goal(map):
    """
    replaces the goal square of the map with a filled square
    """
    map[-1] = map[-1][:-1] + "F"


def to_flat(map, i, j):
    """
    convert `(i,j)` state coordinates to state index in the frozenlake
    `observation_space`.
    """
    return i * len(map) + j


def pre_state_reward(map, state, reward):
    """
    returns a tensor with a value of `reward` in each state action pair
    leading to `state`.
    """
    n = len(map)
    reward_map = torch.zeros((n**2, 4))
    i, j = state
    if i > 0:
        reward_map[to_flat(map, i - 1, j), 1] = reward
    if i < n - 1:
        reward_map[to_flat(map, i + 1, j), 3] = reward
    if j > 0:
        reward_map[to_flat(map, i, j - 1), 2] = reward
    if j < n - 1:
        reward_map[to_flat(map, i, j + 1), 0] = reward
    return reward_map


def constraint_map(map, penalty=2, base_reward=0.1):
    constr_map = torch.zeros((len(map) ** 2, 4))
    for i, row in enumerate(map):
        for j, char in enumerate(row):
            if char == "H":
                constr_map += pre_state_reward(map, (i, j), -penalty - base_reward)
    constr_map += base_reward
    return constr_map


def objective_map(map, goal_reward: float = 10):
    obj_map = torch.zeros((len(map) ** 2, 4))
    for i, row in enumerate(map):
        for j, char in enumerate(row):
            if char == "G":
                obj_map += pre_state_reward(map, (i, j), reward=goal_reward)
    return obj_map


def plot_policy(
    map: list[str],
    agent: Agent,
    n_trajectories: int,
    max_episode_length: int,
    save: bool = False,
    filename: str = None,
    show_goal: str = False,
):
    with torch.no_grad():
        trajs = [
            agent.get_trajectory(max_episode_length) for _ in range(n_trajectories)
        ]
        lambdaHat = util.compute_occupancy_measure(
            trajs, len(map[0]) ** 2, 4, agent.discount_factor
        )
        state_occupancy = np.array(np.array_split(lambdaHat.sum(1).numpy(), len(map)))
    viridis = plt.get_cmap("viridis")
    colors = viridis(state_occupancy)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(colors)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _arrow_col = "gray"

    for j, row in enumerate(map):
        for i, char in enumerate(row):
            if char != "F" and (char != "G" or show_goal):
                ax.text(i, j, char, color="w", size=20, va="center", ha="center")
            if char != "G":
                with torch.no_grad():
                    dist = agent.get_state_dist(to_flat(map, j, i)).probs
                    ax.annotate(
                        "",
                        xytext=(i + 0.1, j),
                        xy=(i + 0.45, j),
                        arrowprops={
                            "width": 10 * dist[2],
                            "headwidth": 10 * dist[2] + 3,
                            "color": _arrow_col,
                        },
                    )
                    ax.annotate(
                        "",
                        xytext=(i - 0.1, j),
                        xy=(i - 0.45, j),
                        arrowprops={
                            "width": 10 * dist[0],
                            "headwidth": 10 * dist[0] + 3,
                            "color": _arrow_col,
                        },
                    )
                    ax.annotate(
                        "",
                        xytext=(i, j + 0.1),
                        xy=(i, j + 0.45),
                        arrowprops={
                            "width": 10 * dist[1],
                            "headwidth": 10 * dist[1] + 3,
                            "color": _arrow_col,
                        },
                    )
                    ax.annotate(
                        "",
                        xytext=(i, j - 0.1),
                        xy=(i, j - 0.45),
                        arrowprops={
                            "width": 10 * dist[3],
                            "headwidth": 10 * dist[3] + 3,
                            "color": _arrow_col,
                        },
                    )
    ax.set_xticks(np.arange(len(map)))
    ax.set_yticks(np.arange(len(map)))
    ax.set_title(r"(Discounted) Occupancy Measure Estimate $\hat{\lambda}^{\pi}(s, a)$")
    fig.tight_layout()

    if save and filename:
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.pdf")

    return (fig, ax)


def trajectory_from_actions(actions, size):
    trajectory = {"actions": actions}
    state, states = 0, [0]
    for action in actions[:-1]:
        match action:
            # left
            case 0:
                if not state % size == 0:
                    state -= 1
            # down
            case 1:
                if not state + size >= size**2:
                    state += size
            # right
            case 2:
                if not (state + 1) % size == 0:
                    state += 1
            # up
            case 3:
                if not state < size:
                    state -= size
        states.append(state)
    trajectory["states"] = states
    return trajectory


def generate_policy_gif(
    experiment: Experiment,
    n_trajectories: int,
    filepath,
    step_interval=1,
    end_frames=10,
    map=None,
):
    frame_files = []
    if map is None:
        map = experiment.save_state["experiment_kwargs"]["env_kwargs"]["desc"]
    for i in range(len(experiment.save_state["checkpoints"]) // step_interval):
        experiment.load_checkpoint(step_interval * i)

        fig, ax = plot_policy(
            map, experiment.agent, n_trajectories, experiment.max_episode_length
        )
        epoch_str = str(experiment.epoch)
        ax.set_title(
            r"(Discounted) Occupancy Measure Estimate $\hat{\lambda}^{\pi}(s,a)$, Epoch: "
            + epoch_str
        )
        frame_files.append(f"policy_imgs/frame_{epoch_str}.png")
        plt.savefig(frame_files[-1])
        plt.close()
    experiment.load_checkpoint()
    fig, ax = plot_policy(
        map, experiment.agent, n_trajectories, experiment.max_episode_length
    )
    epoch_str = str(experiment.epoch)
    ax.set_title(
        r"(Discounted) Occupancy Measure Estimate $\hat{\lambda}^{\pi}(s,a)$, Epoch: "
        + epoch_str
    )
    frame_files += [f"policy_imgs/frame_{epoch_str}.png" for _ in range(end_frames)]
    plt.savefig(frame_files[-1])
    plt.close()

    frames = [Image.open(frame).convert("RGBA") for frame in frame_files]
    frames[0].save(
        filepath, save_all=True, append_images=frames[1:], duration=200, loop=0
    )

    for frame in frame_files:
        try:
            os.remove(frame)
        except:
            continue


def remove_holes(map):
    new_map = []
    for row in map:
        new_map.append("")
        for char in row:
            if char in "SG":
                new_map[-1] += char
            else:
                new_map[-1] += "F"
    return new_map


def plot_environment(
    map: list[str],
    save: bool = False,
    filename: str = None,
    figsize: tuple = (8, 8),
):
    """
    Visualize a grid world environment.

    Args:
        map: List of strings representing the grid world where:
             'S' = Start position
             'F' = Frozen (safe) tile
             'H' = Hole (hazard)
             'G' = Goal
        save: Whether to save the figure to a file
        filename: Filename to save the figure (required if save=True)
        figsize: Figure size as (width, height) tuple

    Returns:
        (fig, ax): Matplotlib figure and axis objects
    """
    n = len(map)

    color_map = {
        "S": [0.2, 0.8, 0.2, 1.0],  # Green for start
        "F": [0.9, 0.9, 0.9, 1.0],  # Light gray for frozen
        "H": [1.0, 0.7, 0.7, 1.0],
        "G": [1.0, 0.84, 0.0, 1.0],  # Gold for goal
    }

    grid_colors = np.zeros((n, n, 4))
    for i, row in enumerate(map):
        for j, char in enumerate(row):
            grid_colors[i, j] = color_map.get(char, [1.0, 1.0, 1.0, 1.0])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid_colors)

    for i, row in enumerate(map):
        for j, char in enumerate(row):
            if char != "F":
                ax.text(
                    j,
                    i,
                    char,
                    color="black" if char != "H" else "white",
                    size=20,
                    weight="bold",
                    va="center",
                    ha="center",
                )

    # Configure axes
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_yticklabels(np.arange(n))
    ax.set_title("Grid World Environment", fontsize=20, weight="bold")

    # Add grid lines
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    fig.tight_layout()

    if save and filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    return (fig, ax)
