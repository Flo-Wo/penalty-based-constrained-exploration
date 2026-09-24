# GridWorld installation

Run from the repository root:

```bash
conda create -n neurips-gridworld python=3.10.19 -y
conda activate neurips-gridworld
python -m pip install -r code/gridworld/requirements.txt
```

Then run one configuration and seed:

```bash
cd code/gridworld
python Penalty_run.py 2_PenaltyAblation.json --task-id 0
```

See the [experiment commands](README.md) and the paper appendix for the experiment settings.
