"""
utils.py - shared helpers (owner: Member C)
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Fix every source of randomness so a run can be repeated exactly (on the same hardware)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     # CPU: weight init, dropout, shuffling
    torch.cuda.manual_seed_all(seed)            # GPU, if present
    torch.backends.cudnn.deterministic = True   # repeatable GPU maths (slightly slower)
    torch.backends.cudnn.benchmark = False


def get_device():
    """'cuda' if an NVIDIA GPU is available, otherwise 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")