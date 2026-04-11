"""
Detector wrapper: sklearn-compatible estimator.
Semi-supervised: trains on normal windows only (no labeled events needed).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ForeignObjectDetector(BaseEstimator, ClassifierMixin):
    """
    Semi-supervised foreign object detector using Isolation Forest.

    Trains on normal (no-event) windows, then flags windows exceeding
    an anomaly threshold as potential foreign object events.

    Parameters
    ----------
    contamination : float        Expected fraction of anomalies.
    n_estimators : int           Trees in the Isolation Forest.
    threshold_quantile : float   Quantile of training scores for the boundary.
    """

    def __init__(self, contamination=0.05, n_estimators=200,
                 threshold_quantile=0.97, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.threshold_quantile = threshold_quantile
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit on normal windows only (semi-supervised)."""
        X_normal = X[y == 0] if y is not None else X
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
            ))
        ])
        self.pipeline_.fit(X_normal)
        train_scores = self.decision_function(X_normal)
        self.threshold_ = float(np.quantile(train_scores, self.threshold_quantile))
        return self

    def decision_function(self, X):
        """Return anomaly scores — higher means more anomalous."""
        return -self.pipeline_.decision_function(X)

    def predict(self, X):
        return (self.decision_function(X) > self.threshold_).astype(int)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        p = 1 / (1 + np.exp(-5 * (scores - self.threshold_)))
        return np.column_stack([1 - p, p])
