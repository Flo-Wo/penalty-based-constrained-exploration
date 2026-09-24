# GridWorld experiments

Follow the [installation instructions](INSTALLATION.md), then run commands from this directory.

| Configuration | Method | Script |
| :--- | :--- | :--- |
| `1_UnconAblation.json` | Unconstrained exploration | `Penalty_run.py` |
| `2_PenaltyAblation.json` | Penalty regularization | `Penalty_run.py` |
| `3_CostAblation.json` | Penalty regularization with a smaller hole cost | `Penalty_run.py` |
| `4_VRPDPG.json` | VR-PDPG | `VRPDPG_run.py` |
| `5_PDPG.json` | PDPG | `PDPG_run.py` |
| `6_Norm_constr.json` | Reference-occupancy norm constraint | `Norm_constr_run.py` |

For example, run one configuration and seed from the penalty sweep:

```bash
python Penalty_run.py 2_PenaltyAblation.json --task-id 0
```

Use `--count-tasks` to print the number of configuration/seed pairs. Task IDs are zero-based; omit `--task-id` to run the full sweep locally. Each JSON file sets the hyperparameters, number of seeds, training duration, and output `PATH`. See the paper appendix for the experimental details.

The makefile provides a shortcut for each supplied configuration:

```bash
make ablation_penalty TASK_ID=0
```

The other targets are `ablation_unconstrained`, `ablation_penalty_smaller_costs`, `ablation_VRPDPG`, `ablation_PDPG`, and `ablation_norm_constraint`. Omit `TASK_ID` to run a full sweep. These targets run Python locally and do not require a cluster scheduler.
