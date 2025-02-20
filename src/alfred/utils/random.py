import numpy as np
import torch

import random
import os

from alfred.devices import set_device



def set_deterministic(seed=0):  # Add device as an argument
    """Sets random seeds for reproducibility (CPU, CUDA, or MPS)."""

    device = set_device()

    # 1. Python's `random` module
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)

    # 4. Device-specific seeding
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == 'mps':
        torch.mps.manual_seed(seed)  # MPS seeding

    # 5. Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
