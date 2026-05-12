"""
probe.py — Scaler -> PCA(pooled) -> Scaler(geo) -> LogisticRegression.

Keeps the public HallucinationProbe API expected by solution.py while
remaining lightweight and fully within the editable-file budget of the task.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn


class HallucinationProbe(nn.Module):
    """Sklearn-based probe matching the required public API.

    The probe reduces the pooled hidden-state block to 20 PCA components,
    keeps the six geometric features on a separate standardized scale, and
    trains a regularized logistic regression classifier.
    """

    def __init__(self, geo_dim: int = 6, n_components: int = 20) -> None:
        super().__init__()
        self.scaler_p = StandardScaler()
        self.scaler_g = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)

        self.clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        self._threshold: float = 0.5
        self._geo_dim: int = geo_dim
        self._n_components: int = n_components
        self._is_fitted: bool = False

    def _process(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        p_dim = X.shape[1] - self._geo_dim
        X_p, X_g = X[:, :p_dim], X[:, p_dim:]
        
        X_p_sc = self.scaler_p.fit_transform(X_p) if fit else self.scaler_p.transform(X_p)
        X_p_pca = self.pca.fit_transform(X_p_sc) if fit else self.pca.transform(X_p_sc)
        
        if self._geo_dim > 0:
            X_g_sc = self.scaler_g.fit_transform(X_g) if fit else self.scaler_g.transform(X_g)
            return np.hstack([X_p_pca, X_g_sc])
        return X_p_pca

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_final = self._process(X, fit=True)
        self.clf.fit(X_final, y)
        self._is_fitted = True
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_acc = 0.5, -1.0
        for t in candidates:
            acc = accuracy_score(y_val, (probs >= t).astype(int))
            if (acc > best_acc) or (
                np.isclose(acc, best_acc) and abs(float(t) - 0.5) < abs(best_t - 0.5)
            ):
                best_acc, best_t = float(acc), float(t)
        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X_final = self._process(X, fit=False)
        prob_pos = self.clf.predict_proba(X_final)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
