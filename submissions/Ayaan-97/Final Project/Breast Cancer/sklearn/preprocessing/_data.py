from __future__ import annotations

import numpy as np


class StandardScaler:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def transform(self, X, copy=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if getattr(self, "with_mean", True) and hasattr(self, "mean_"):
            X = X - np.asarray(self.mean_, dtype=float)

        if getattr(self, "with_std", True) and hasattr(self, "scale_"):
            scale = np.asarray(self.scale_, dtype=float)
            scale = np.where(scale == 0, 1.0, scale)
            X = X / scale

        if copy:
            return X.copy()
        return X
