import torch, numpy as np, random, os


def set_seed(seed: int = 42) -> None:
    """Sets the seeds for PyTorch, NumPy, and Python's random module."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, these two options must be set for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
