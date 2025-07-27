"""
Random Forest Classifier (Manual Implementation)
------------------------------------------------
An ensemble of manually implemented Decision Trees for binary classification.
Uses bootstrapped samples and random feature selection at each split.
"""

import numpy as np
from typing import List, Tuple, Optional
from models import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Custom Random Forest Classifier for binary classification.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree.
    min_samples_split : int
        Minimum number of samples to split a node.
    max_features : int
        Number of features to consider for each split (random subset).
    bootstrap : bool
        Whether to sample with replacement for each tree.
    random_state : int
        Seed for reproducibility.
    """
    def __init__(self, n_estimators: int = 10, max_depth: int = 5, min_samples_split: int = 2,
                 max_features: Optional[int] = None, bootstrap: bool = True, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifier] = []
        self.features_indices: List[np.ndarray] = []
        np.random.seed(self.random_state)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a bootstrap sample for one tree."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Fit the Random Forest."""
        n_samples, n_features = X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))  # Standard RF heuristic

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = (self._bootstrap_sample(X, y) if self.bootstrap else (X, y))
            # Random feature selection
            feat_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X_sample[:, feat_indices]

            # Train manual decision tree on subset
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_sample)

            self.trees.append(tree)
            self.features_indices.append(feat_indices)

        return self

    def _predict_one(self, x: np.ndarray) -> int:
        """Aggregate predictions (majority vote) from all trees."""
        votes = []
        for tree, feat_idx in zip(self.trees, self.features_indices):
            x_subset = x[feat_idx]
            votes.append(tree._predict_one(x_subset))
        return int(np.round(np.mean(votes)))  # Majority vote

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for multiple samples."""
        return np.array([self._predict_one(row) for row in X])