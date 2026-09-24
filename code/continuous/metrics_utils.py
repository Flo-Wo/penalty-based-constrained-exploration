"""
Utilities for logging and loading training metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, save_dir: str):
        """
        Initialize metrics logger.

        Args:
            save_dir: Directory to save metrics
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics = {
            "iteration": [],
            "entropy": [],
            "mean_constraint": [],
            "constraint_violation": [],
            "penalty": [],
            "occ_loss": [],
            "policy_loss": [],
            "critic_loss": [],
            "beta": [],
            "mean_reward": [],
        }

        self.csv_path = os.path.join(save_dir, "metrics.csv")
        self.json_path = os.path.join(save_dir, "metrics.json")

    def log(self, iteration: int, **kwargs):
        """
        Log metrics for current iteration.

        Args:
            iteration: Current iteration number
            **kwargs: Metric values (entropy, mean_constraint, etc.)
        """
        self.metrics["iteration"].append(iteration)

        for key in [
            "entropy",
            "mean_constraint",
            "constraint_violation",
            "penalty",
            "occ_loss",
            "policy_loss",
            "critic_loss",
            "beta",
            "mean_reward",
        ]:
            value = kwargs.get(key, np.nan)
            self.metrics[key].append(float(value) if value is not None else np.nan)

    def save(self):
        """Save metrics to CSV and JSON files."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_path, index=False)

        with open(self.json_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics(self) -> Dict[str, List[float]]:
        """Return current metrics dictionary."""
        return self.metrics.copy()


def load_metrics(checkpoint_dir: str) -> Dict[str, np.ndarray]:
    """
    Load training metrics from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing metrics files

    Returns:
        Dictionary with metric arrays
    """
    csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    json_path = os.path.join(checkpoint_dir, "metrics.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        metrics = {col: df[col].values for col in df.columns}
        return metrics

    elif os.path.exists(json_path):
        with open(json_path, "r") as f:
            metrics = json.load(f)
        metrics = {k: np.array(v) for k, v in metrics.items()}
        return metrics

    else:
        raise FileNotFoundError(f"No metrics found in {checkpoint_dir}")


def plot_training_metrics(checkpoint_dir: str, save_path: Optional[str] = None):
    """
    Plot training metrics from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing metrics files
        save_path: Optional path to save plot (if None, uses checkpoint_dir/training_curves.png)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot plot metrics")
        return

    metrics = load_metrics(checkpoint_dir)
    iterations = metrics["iteration"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    plot_configs = [
        ("entropy", "Entropy (H)", "Entropy"),
        ("mean_constraint", "Mean Constraint E[g]", "Constraint Value"),
        ("constraint_violation", "Constraint Violation (E[g]+)", "Violation"),
        ("penalty", "Penalty", "Penalty"),
        ("occ_loss", "Occupancy Loss (NLL)", "Loss"),
        ("policy_loss", "Policy Loss", "Loss"),
        ("critic_loss", "Critic Loss", "Loss"),
        ("beta", "Beta (Penalty Weight)", "Beta"),
        ("mean_reward", "Mean Episode Reward", "Reward"),
    ]

    for idx, (key, title, ylabel) in enumerate(plot_configs):
        ax = axes[idx]
        if key in metrics:
            values = metrics[key]
            mask = ~np.isnan(values)
            ax.plot(iterations[mask], values[mask], alpha=0.7, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            if key in ["mean_constraint", "constraint_violation"]:
                ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(checkpoint_dir, "training_curves.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_specific_metrics(
    checkpoint_dir: str,
    metric_keys: List[str],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot specific metrics from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing metrics files
        metric_keys: List of metric keys to plot
        save_path: Optional path to save plot
        title: Optional title for the plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot plot metrics")
        return

    metrics = load_metrics(checkpoint_dir)
    iterations = metrics["iteration"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for key in metric_keys:
        if key in metrics:
            values = metrics[key]
            mask = ~np.isnan(values)
            ax.plot(iterations[mask], values[mask], alpha=0.7, linewidth=2, label=key)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(checkpoint_dir, f'metrics_{"_".join(metric_keys)}.png')

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def compare_runs(
    checkpoint_dirs: List[str],
    metric_key: str = "mean_constraint",
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Compare a specific metric across multiple runs.

    Args:
        checkpoint_dirs: List of checkpoint directories to compare
        metric_key: Metric to compare
        labels: Optional labels for each run
        save_path: Optional path to save plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot plot metrics")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if labels is None:
        labels = [os.path.basename(d) for d in checkpoint_dirs]

    for checkpoint_dir, label in zip(checkpoint_dirs, labels):
        try:
            metrics = load_metrics(checkpoint_dir)
            iterations = metrics["iteration"]
            values = metrics[metric_key]

            mask = ~np.isnan(values)
            ax.plot(iterations[mask], values[mask], alpha=0.7, linewidth=2, label=label)
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_dir}: {e}")

    ax.set_title(f"Comparison: {metric_key}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_key)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = f"comparison_{metric_key}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def print_summary(checkpoint_dir: str):
    """
    Print summary statistics of training run.

    Args:
        checkpoint_dir: Directory containing metrics files
    """
    metrics = load_metrics(checkpoint_dir)

    print(f"\n{'='*60}")
    print(f"Training Summary: {checkpoint_dir}")
    print(f"{'='*60}")

    for key in [
        "entropy",
        "mean_constraint",
        "constraint_violation",
        "penalty",
        "policy_loss",
        "critic_loss",
        "mean_reward",
    ]:
        if key in metrics:
            values = metrics[key]
            values = values[~np.isnan(values)]

            if len(values) > 0:
                print(f"\n{key}:")
                print(f"  Final: {values[-1]:.4f}")
                print(f"  Mean:  {np.mean(values):.4f}")
                print(f"  Std:   {np.std(values):.4f}")
                print(f"  Min:   {np.min(values):.4f}")
                print(f"  Max:   {np.max(values):.4f}")

    print(f"\n{'='*60}\n")
