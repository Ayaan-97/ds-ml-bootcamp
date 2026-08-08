from __future__ import annotations

import numpy as np


class SVC:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def _rbf_kernel(self, X, Y):
        gamma = getattr(self, "_gamma", None)
        if gamma is None:
            gamma = 1.0 / X.shape[1] if X.shape[1] else 1.0

        X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]
        squared_distances = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
        return np.exp(-gamma * squared_distances)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        support_vectors = np.asarray(self.support_vectors_, dtype=float)
        dual_coef = np.asarray(self.dual_coef_, dtype=float)
        intercept = np.asarray(self.intercept_, dtype=float)

        kernel_matrix = self._rbf_kernel(X, support_vectors)
        return np.dot(kernel_matrix, dual_coef.T).ravel() + intercept[0]

    def predict(self, X):
        decision_values = self.decision_function(X)
        classes = np.asarray(self.classes_)
        if classes.size != 2:
            raise NotImplementedError("Only binary SVC predictions are supported")
        return np.where(decision_values > 0, classes[1], classes[0])
