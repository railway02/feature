from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


_CORE: Any = None
_MODE = ""


@dataclass
class DeepBranch:
    requested_dim: int
    seed: int
    scaler: StandardScaler | None = None
    pca: PCA | None = None

    def fit(self, values: np.ndarray) -> "DeepBranch":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise AssertionError(f"Deep branch must be finite 2-D, got {values.shape}")
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(values)
        n_components = max(
            1,
            min(int(self.requested_dim), values.shape[0] - 1, values.shape[1]),
        )
        self.pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=self.seed,
        )
        self.pca.fit(scaled)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Deep branch is not fitted")
        output = self.pca.transform(self.scaler.transform(values)).astype(np.float32)
        if not np.isfinite(output).all():
            raise AssertionError("Deep branch transform is nonfinite")
        return output

    def audit(self) -> dict[str, Any]:
        if self.pca is None:
            return {}
        return {
            "requested_pca_dimension": int(self.requested_dim),
            "fitted_pca_dimension": int(self.pca.n_components_),
            "explained_variance_ratio_sum": float(
                np.sum(self.pca.explained_variance_ratio_)
            ),
        }


@dataclass
class MatchedPreprocessor:
    deep_pca_dim: int
    use_scalar: bool
    scalar_pca_dim: int = 48
    seed: int = 20260804

    primary: DeepBranch | None = None
    secondary: DeepBranch | None = None
    morphology: Any = None
    deep_pca: PCA | None = None

    def fit(self, deep: np.ndarray, scalar: np.ndarray) -> "MatchedPreprocessor":
        if _MODE == "M0":
            if self.use_scalar:
                raise AssertionError("M0 must not enable a second branch")
            self.morphology = _CORE.ScalarPreprocessor().fit(
                deep,
                pca_dim=self.deep_pca_dim,
                seed=self.seed,
            )
            self.deep_pca = self.morphology.pca
        elif _MODE == "W0":
            if self.use_scalar:
                raise AssertionError("W0 must not enable a second branch")
            self.primary = DeepBranch(self.deep_pca_dim, self.seed).fit(deep)
            self.deep_pca = self.primary.pca
        elif _MODE == "WL":
            if not self.use_scalar:
                raise AssertionError("WL requires independent Whole and Local branches")
            self.primary = DeepBranch(self.deep_pca_dim, self.seed).fit(deep)
            self.secondary = DeepBranch(self.deep_pca_dim, self.seed + 17).fit(
                scalar
            )
            self.deep_pca = self.primary.pca
        else:
            raise ValueError(f"Unknown matched experiment mode: {_MODE}")
        return self

    def transform(self, deep: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        if _MODE == "M0":
            if self.morphology is None:
                raise RuntimeError("M0 morphology preprocessor is not fitted")
            output = self.morphology.transform(deep)
        elif _MODE == "W0":
            if self.primary is None:
                raise RuntimeError("W0 Whole branch is not fitted")
            output = self.primary.transform(deep)
        elif _MODE == "WL":
            if self.primary is None or self.secondary is None:
                raise RuntimeError("WL branches are not fitted")
            output = np.concatenate(
                [self.primary.transform(deep), self.secondary.transform(scalar)],
                axis=1,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown matched experiment mode: {_MODE}")
        if not np.isfinite(output).all():
            raise AssertionError("Matched preprocessor output is nonfinite")
        return output

    def audit(self) -> dict[str, Any]:
        if _MODE == "M0":
            return {
                "experiment": _MODE,
                "input": "Mask morphology only",
                "morphology": self.morphology.audit() if self.morphology else {},
                "valid_fit_forbidden": True,
            }
        if _MODE == "W0":
            return {
                "experiment": _MODE,
                "input": "Whole deep only",
                "whole": self.primary.audit() if self.primary else {},
                "valid_fit_forbidden": True,
            }
        return {
            "experiment": _MODE,
            "input": "Whole and Local independently scaled/PCA then concatenated",
            "whole": self.primary.audit() if self.primary else {},
            "local": self.secondary.audit() if self.secondary else {},
            "separate_branch_fit": True,
            "valid_fit_forbidden": True,
        }


def matched_slice_components(
    transformed: np.ndarray,
    requested_deep_dim: int,
    fitted_deep_dim: int,
    use_scalar: bool,
) -> np.ndarray:
    transformed = np.asarray(transformed)
    primary_count = min(int(requested_deep_dim), int(fitted_deep_dim))
    if _MODE in {"M0", "W0"}:
        return transformed[:, :primary_count]
    if _MODE != "WL" or not use_scalar:
        raise AssertionError(f"Unexpected slice contract: mode={_MODE}, scalar={use_scalar}")
    secondary_fitted = transformed.shape[1] - int(fitted_deep_dim)
    secondary_count = min(int(requested_deep_dim), secondary_fitted)
    return np.concatenate(
        [
            transformed[:, :primary_count],
            transformed[
                :,
                int(fitted_deep_dim) : int(fitted_deep_dim) + secondary_count,
            ],
        ],
        axis=1,
    )


def install(core: Any, mode: str) -> None:
    global _CORE, _MODE
    if mode not in {"M0", "W0", "WL"}:
        raise ValueError(mode)
    _CORE = core
    _MODE = mode
    core.FusionPreprocessor = MatchedPreprocessor
    core.slice_fusion_components = matched_slice_components

