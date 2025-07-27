"""
Custom Decision Tree Classifier
------------------------------------------------
Binary classification tree for loan approval prediction.
Uses Gini Impurity for splits and supports recursive tree growth.
"""

import numpy as np
from typing import Optional, Tuple

class DecisionTreeNode:
    """Node class for the decision tree."""
    def __init__(self, gini: float, num_samples: int, num_samples_per_class: np.ndarray,
                 predicted_class: int):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional['DecisionTreeNode'] = None
        self.right: Optional['DecisionTreeNode'] = None

class DecisionTreeClassifier:
    """
    Custom Decision Tree Classifier for binary classification.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_split : int
        Minimum samples to split a node.
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.tree_: Optional[DecisionTreeNode] = None

    def _gini(self, y: np.ndarray) -> float:
        """Compute Gini impurity of a label array."""
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / m
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find best split (feature, threshold) for a node."""
        m, n = X.shape
        if m <= 1:
            return None, None

        best_gini = 1.0
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = np.bincount(classes, minlength=self.n_classes_).tolist()
            for i in range(1, m):  # split at each unique value
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((nl / i) ** 2 for nl in num_left if i > 0)
                gini_right = 1.0 - sum((nr / (m - i)) ** 2 for nr in num_right if (m - i) > 0)
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        num_samples_per_class = np.bincount(y, minlength=self.n_classes_)
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth and len(y) >= self.min_samples_split and node.gini > 0:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """Fit the decision tree to the data."""
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        return self

    def _predict_one(self, inputs: np.ndarray) -> int:
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        return np.array([self._predict_one(row) for row in X])
