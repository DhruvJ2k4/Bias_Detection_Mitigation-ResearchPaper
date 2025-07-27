import logging
import random
import numpy as np
import yaml
from typing import Any, Dict

def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across NumPy and Python random."""
    random.seed(seed)
    np.random.seed(seed)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger for consistent output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger
