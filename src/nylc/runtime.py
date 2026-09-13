"""Explicit CPU/GPU choices; changing hardware never changes model identity."""

import os
import random
import numpy as np


def setup_runtime(config):
    count = str(config["runtime"]["threads"])
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = count
    os.environ.setdefault("MPLBACKEND", "Agg")
    random.seed(config["gp"]["seed"])
    np.random.seed(config["gp"]["seed"])


def setup_torch(runtime):
    if runtime["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    device = runtime["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; choose runtime.device=cpu explicitly"
        )
    torch.manual_seed(runtime["seed"])
    torch.set_num_threads(runtime["threads"])
    torch.use_deterministic_algorithms(runtime["deterministic"])
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = runtime["deterministic"]
    return device
