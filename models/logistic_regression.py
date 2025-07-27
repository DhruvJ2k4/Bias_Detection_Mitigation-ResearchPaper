"""
Logistic Regression Baseline Model for Loan Approval Bias Detection
--------------------------------------------------------------------
Implements a reproducible, extensible Logistic Regression classifier.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from typing import Tuple, Dict

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 500
) -> Tuple[BaseEstimator, np.ndarray, np.ndarray]:
    """
    Train a Logistic Regression classifier for loan approval prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Preprocessed test features.
    y_test : np.ndarray
        Test labels.
    random_state : int, optional
        Seed for reproducibility.
    C : float, optional
        Inverse regularization strength.
    max_iter : int, optional
        Maximum iterations for solver convergence.

    Returns
    -------
    model : BaseEstimator
        Trained Logistic Regression model.
    y_pred : np.ndarray
        Predicted class labels for test set.
    y_prob : np.ndarray
        Predicted probabilities for positive class (test set).
    """
    model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_prob
