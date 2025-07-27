"""
Manual Gradient Boosting Classifier
-----------------------------------
Simplified gradient boosting for binary classification (logistic loss).
Inspired by XGBoost, implemented in pure Python/NumPy for research pipelines.
Uses shallow manual decision trees as base learners.
"""

import numpy as np
from typing import List
from models import DecisionTreeClassifierManual  # Reuse our manual tree

class GradientBoostingClassifier:
    """
    Custom Gradient Boosting Classifier.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage factor for each tree's contribution.
    max_depth : int
        Depth of base learners (trees).
    min_samples_split : int
        Minimum samples to split nodes.
    random_state : int
        Seed for reproducibility.
    """
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.trees: List[DecisionTreeClassifierManual] = []
        self.gammas: List[float] = []  # weights for each tree

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation for probability output."""
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """
        Fit the gradient boosting model using logistic loss.
        y must be binary (0 or 1).
        """
        # Initialize predictions with log odds (prior)
        y_mean = np.mean(y)
        log_odds = np.log(y_mean / (1 - y_mean + 1e-12))
        self.F0 = log_odds  # initial bias
        F = np.full(y.shape, self.F0)

        for _ in range(self.n_estimators):
            # Compute pseudo-residuals (negative gradient of logistic loss)
            p = self._sigmoid(F)
            residual = y - p  # gradient

            # Fit a tree to residuals
            tree = DecisionTreeClassifierManual(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, (residual > 0).astype(int))  # classify residual signs (approximation)
            
            # Get tree predictions (+1/-1 for gradient sign)
            pred = tree.predict(X) * 2 - 1

            # Compute optimal step size (gamma) via line search (simplified)
            gamma = np.sum(residual * pred) / (np.sum(np.abs(pred)) + 1e-12)

            # Update F
            F += self.learning_rate * gamma * pred

            # Store tree and weight
            self.trees.append(tree)
            self.gammas.append(gamma)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class."""
        F = np.full(X.shape[0], self.F0)
        for tree, gamma in zip(self.trees, self.gammas):
            pred = tree.predict(X) * 2 - 1
            F += self.learning_rate * gamma * pred
        return self._sigmoid(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int)
