import numpy as np
import pandas as pd

def statistical_parity_difference(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """Difference in positive prediction rates between protected and unprotected groups."""
    return np.mean(y_pred[protected == 1]) - np.mean(y_pred[protected == 0])

def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, protected: np.ndarray) -> float:
    """Difference in true positive rates between protected and unprotected groups."""
    def tpr(y_t, y_p): return np.sum((y_p == 1) & (y_t == 1)) / np.sum(y_t == 1)
    return tpr(y_true[protected == 1], y_pred[protected == 1]) - tpr(y_true[protected == 0], y_pred[protected == 0])

def disparate_impact_ratio(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """Ratio of positive outcome rates (protected/unprotected)."""
    p_rate = np.mean(y_pred[protected == 1]) / np.mean(y_pred[protected == 0])
    return p_rate
