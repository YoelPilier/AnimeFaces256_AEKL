import random
import numpy as np
import torch


def fix_seed(seed: int = 65535, deterministic: bool = False):
    """
    Fix random seeds for reproducibility.

    Args:
        seed: Random seed.
        deterministic: Enable deterministic algorithms.
                       May reduce performance.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return seed
