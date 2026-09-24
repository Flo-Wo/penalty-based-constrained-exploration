# Continuous-control installation

Run from the repository root:

```bash
conda create -n neurips-control python=3.11.14 -y
conda activate neurips-control
python -m pip install -r code/continuous/requirements.txt
```

Video rendering requires OpenGL and FFmpeg. If needed, install FFmpeg in the active environment:

```bash
conda install -c conda-forge ffmpeg
```

See the [main README](../../README.md#running-the-experiments) for the SafeCartpole and PointMass commands, and the paper appendix for the experiment settings.
