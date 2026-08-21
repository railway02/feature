from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class TrainFeaturePreprocessor:
    columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    add_missing_indicators: bool = True
    missing_indicator_columns: list[str] | None = None

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str], add_missing_indicators: bool = True):
        medians, means, stds = {}, {}, {}
        missing_indicator_columns = []
        for c in columns:
            x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = float(x.median()) if x.notna().any() else 0.0
            z = x.fillna(med).astype(float)
            mean = float(z.mean())
            std = float(z.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            medians[c] = med; means[c] = mean; stds[c] = std
            if add_missing_indicators and x.isna().any():
                missing_indicator_columns.append(c)
        return cls(columns=list(columns), medians=medians, means=means, stds=stds,
                   add_missing_indicators=bool(add_missing_indicators),
                   missing_indicator_columns=missing_indicator_columns)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        parts, names = [], []
        for c in self.columns:
            x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            miss = x.isna().to_numpy(dtype=np.float32)
            z = x.fillna(self.medians[c]).to_numpy(dtype=np.float32)
            z = (z - self.means[c]) / self.stds[c]
            parts.append(z[:, None]); names.append(c)
            indicator_columns = self.missing_indicator_columns
            # Backward compatibility for old saved preprocessors.
            if indicator_columns is None:
                indicator_columns = self.columns
            if self.add_missing_indicators and c in indicator_columns:
                parts.append(miss[:, None]); names.append(f"{c}__missing")
        if not parts:
            return np.zeros((len(df), 0), np.float32), []
        out = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("Non-finite feature remained after TrainFeaturePreprocessor.transform")
        return out, names

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({
            "columns": self.columns,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "add_missing_indicators": self.add_missing_indicators,
            "missing_indicator_columns": self.missing_indicator_columns,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)
