"""Global seeding for reproducible runs.

``torch`` is seeded only if it is already importable, so this helper stays usable
in lightweight environments that do not have the full ML stack installed.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if available) PyTorch RNGs.

    Args:
        seed: The seed value.
        deterministic: If True and torch+CUDA are present, request deterministic
            cuDNN behaviour (slower but reproducible).

    Returns:
        The seed that was applied (for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional in light environments
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed
