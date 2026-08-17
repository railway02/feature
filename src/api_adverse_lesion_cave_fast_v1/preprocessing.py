from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class DeepOnlyPreprocessor:
    branch: object | None = None
    final_scaler: StandardScaler | None = None

    def fit(self, deep: np.ndarray, missing: np.ndarray, seed: int, fixed) -> "DeepOnlyPreprocessor":
        self.branch = fixed.DeepBranch().fit(deep, seed)
        base = np.concatenate([self.branch.transform(deep), missing.astype(np.float64)], axis=1)
        self.final_scaler = StandardScaler().fit(base)
        return self

    def transform(self, deep: np.ndarray, missing: np.ndarray) -> np.ndarray:
        if self.branch is None or self.final_scaler is None:
            raise RuntimeError("Preprocessor not fitted")
        base = np.concatenate([self.branch.transform(deep), missing.astype(np.float64)], axis=1)
        output = self.final_scaler.transform(base).astype(np.float64)
        if not np.isfinite(output).all():
            raise AssertionError("Nonfinite deep features")
        return output


@dataclass
class MorphologyPreprocessor:
    branch: object | None = None
    final_scaler: StandardScaler | None = None

    def fit(self, values: np.ndarray, seed: int, fixed) -> "MorphologyPreprocessor":
        self.branch = fixed.NumericBranch(32, "robust", 0.25, True).fit(values, seed)
        reduced = self.branch.transform(values)
        self.final_scaler = StandardScaler().fit(reduced)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.branch is None or self.final_scaler is None:
            raise RuntimeError("Morphology preprocessor not fitted")
        output = self.final_scaler.transform(self.branch.transform(values)).astype(np.float64)
        if not np.isfinite(output).all():
            raise AssertionError("Nonfinite morphology features")
        return output
