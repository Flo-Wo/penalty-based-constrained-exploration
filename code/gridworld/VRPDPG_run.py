from experiments import Experiment, already_trained

from agent import LinearSoftmaxPolicy, VRPDPG
import general_utility as util
import frozenlake_utility as fl_util

import torch
from itertools import product
from tqdm import tqdm
import json
import sys
import os
import argparse

from utils import set_seed


def _build_hparam_selections(experiment_info: dict) -> tuple[list[str], list[dict]]:
    iter_keys, iter_vals = [], []
    fixed_dict = {}
    for key, val in experiment_info["hyperparameters"].items():
        if isinstance(val, list):
            iter_keys.append(key)
            iter_vals.append(val)
        else:
            fixed_dict[key] = val

    hparams_selections = [dict(zip(iter_keys, vals)) for vals in product(*iter_vals)]
    for d in hparams_selections:
        d.update(fixed_dict)
    return iter_keys, hparams_selections


def _task_to_indices(task_id: int, n_hparam_selections: int, n_experiments: int):
    total_tasks = n_hparam_selections * n_experiments
    if task_id < 0 or task_id >= total_tasks:
        raise ValueError(f"task_id must be in [0, {total_tasks - 1}], got {task_id}")
    hparam_index = task_id // n_experiments
    repetition_index = task_id % n_experiments
    return hparam_index, repetition_index


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Penalty experiments. Supports SLURM array parallelization by "
            "selecting a single (hyperparameter choice, repetition) via SLURM_ARRAY_TASK_ID."
        )
    )
    parser.add_argument(
        "config",
        help="Path to the experiment json config (e.g. ./Penalty_param.json)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help=(
            "Run exactly one task (0-indexed). If omitted, uses SLURM_ARRAY_TASK_ID when set. "
            "If neither is provided, runs all tasks sequentially."
        ),
    )
    parser.add_argument(
        "--one-indexed",
        action="store_true",
        help="Interpret --task-id (or SLURM_ARRAY_TASK_ID) as 1-indexed.",
    )
    parser.add_argument(
        "--count-tasks",
        action="store_true",
        help="Print total number of tasks (len(grid) * n_experiments) and exit.",
    )
    return parser.parse_args(argv)


args = _parse_args(sys.argv[1:])

with open(args.config) as json_file:
    experiment_info = json.load(json_file)
PATH = experiment_info["PATH"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# safe exploration
env_map = ["SFFFFF", "FFFFFF", "FFHHFF", "FFHHFF", "FFFFFF", "FFFFFG"]

iter_keys, hparams_selections = _build_hparam_selections(experiment_info)
n_experiments = int(experiment_info["n_experiments"])
total_tasks = len(hparams_selections) * n_experiments

if args.count_tasks:
    print(total_tasks)
    raise SystemExit(0)

eval_cfg = experiment_info.get("evaluation", {})
eval_enabled = bool(eval_cfg.get("save", True))
eval_force = bool(eval_cfg.get("force", False))
eval_n_trajectories = int(eval_cfg.get("n_trajectories", eval_cfg.get("n_episodes", 8)))

env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
task_id = args.task_id
if task_id is None and env_task_id is not None:
    task_id = int(env_task_id)
if task_id is not None and args.one_indexed:
    task_id -= 1

if task_id is None:
    task_ids = list(range(total_tasks))
    task_iterator = tqdm(task_ids, leave=False)
else:
    task_ids = [task_id]
    task_iterator = task_ids

base_seed = int(experiment_info.get("seed", 0))

for global_task_id in task_iterator:
    hparam_index, k = _task_to_indices(
        global_task_id, len(hparams_selections), n_experiments
    )
    hparams = hparams_selections[hparam_index]

    if task_id is None:
        desc = []
        for key in iter_keys:
            desc.append(f"{key}: {hparams[key]}")
        task_iterator.set_description(", ".join(desc))

    set_seed(base_seed + int(global_task_id))

    base_name = experiment_info["name"].format(**hparams)
    name = base_name + f"__n_{k}"

    run_path = os.path.join(PATH, name)
    eval_path = os.path.join(PATH, f"eval_curve__{name}.pt")

    trained = already_trained(run_path, experiment_info["n_epochs"])
    evaluated = bool(eval_enabled) and os.path.exists(eval_path) and (not eval_force)

    if trained and evaluated:
        print(f"\nSkipping already trained+evaluated experiment: {name}")
        continue

    # the default experiment uses -50, but we have a modified version
    _penalty = hparams.get("hole_penalty", -50)

    experiment = Experiment(
        name=name,
        agent=VRPDPG,
        objective_func=util.Entropy,
        objective_func_kwargs={},
        constraint_func=util.NegativeLinear,
        constraint_func_kwargs={
            "map": fl_util.constraint_map(
                env_map,
                penalty=_penalty,
                base_reward=0.1,
            ).to(device)
        },
        agent_kwargs={
            "discount_factor": hparams["discount_factor"],
            "dual_lr": hparams["dual_lr"],
            "alpha_t": hparams["alpha_t"],
        },
        env_id="FrozenLake-v1",
        env_kwargs={
            "desc": fl_util.remove_holes(env_map),
            "is_slippery": False,
        },
        policy=LinearSoftmaxPolicy,
        policy_kwargs={"n": len(env_map) ** 2, "m": 4},
        optimizer=torch.optim.SGD,
        optimizer_kwargs={"lr": hparams["learning_rate"]},
        max_episode_length=25,
        custom_path=PATH,
        checkpoint_step=100,
        episodes_per_epoch=hparams["episodes_per_epoch"],
    )

    experiment.load_from_file()

    # 1) Train if needed
    if not trained:
        experiment.run_until(experiment_info["n_epochs"])

    # 2) Evaluate all checkpoints if needed
    if eval_enabled and (eval_force or (not os.path.exists(eval_path))):
        # evals is (n_checkpoints, 3): [epoch, obj, constr]
        evals = experiment.eval_run(eval_n_trajectories)
        payload = {
            "name": name,
            "base_name": base_name,
            "hparams": hparams,
            "task_id": int(global_task_id),
            "n_eval_trajectories": int(eval_n_trajectories),
            "epoch": int(experiment.epoch),
            "evals": evals,
        }
        os.makedirs(PATH, exist_ok=True)
        tmp_path = eval_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, eval_path)
