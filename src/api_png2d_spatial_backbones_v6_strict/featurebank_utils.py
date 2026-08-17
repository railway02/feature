from __future__ import annotations

import numpy as np


def select_oof_by_fold(values_by_fold: np.ndarray, outer_fold: np.ndarray) -> np.ndarray:
    values = np.asarray(values_by_fold)
    folds = np.asarray(outer_fold).astype(int)
    if values.ndim < 2 or values.shape[0] != len(folds) or values.shape[1] != 5:
        raise ValueError(f"Expected values [N,5,...] and folds [N], got {values.shape}/{folds.shape}")
    if not np.isin(folds, [1, 2, 3, 4, 5]).all():
        raise ValueError("outer_fold must contain only 1..5")
    return values[np.arange(len(folds)), folds - 1]
