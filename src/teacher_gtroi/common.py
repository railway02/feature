from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(p)
    return cfg


def resolve_path(value: str | Path, project_root: str | Path | None = None) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute() and project_root is not None:
        p = Path(project_root).expanduser() / p
    return p.resolve()


def atomic_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(tmp, path)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def normalize_id(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return str(int(s)) if s.isdigit() else s


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_temporal(cfg: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    path = (
        cfg["data"]["temporal_train_npz"]
        if split == "Train"
        else cfg["data"]["temporal_valid_npz"]
    )
    raw = load_npz(path)
    required = {"deep", "target", "series_uid", "patient_id"}
    missing = required - set(raw)
    if missing:
        raise KeyError(f"{path}: missing keys {sorted(missing)}")

    raw["deep"] = np.asarray(raw["deep"], dtype=np.float32)
    raw["target"] = np.asarray(raw["target"], dtype=np.int64)
    raw["series_uid"] = raw["series_uid"].astype(str)
    raw["patient_id"] = np.asarray([normalize_id(x) for x in raw["patient_id"]], dtype=str)
    return raw


def load_train_folds(cfg: dict[str, Any], train: dict[str, np.ndarray]) -> np.ndarray:
    if "fold" in train:
        folds = np.asarray(train["fold"]).astype(int)
        if len(folds) == len(train["target"]) and set(np.unique(folds)) == {1,2,3,4,5}:
            return folds

    path = str(cfg["data"].get("train_oof_csv_optional", "") or "").strip()
    if not path:
        raise RuntimeError(
            "No valid fold array in train NPZ and train_oof_csv_optional is empty."
        )
    df = pd.read_csv(path)
    if not {"series_uid", "fold"} <= set(df.columns):
        raise KeyError("OOF CSV must contain series_uid and fold")
    mapping = dict(zip(df["series_uid"].astype(str), pd.to_numeric(df["fold"]).astype(int)))
    folds = np.asarray([mapping[u] for u in train["series_uid"]], dtype=int)
    if set(np.unique(folds)) != {1,2,3,4,5}:
        raise AssertionError(f"Unexpected folds: {np.unique(folds)}")
    return folds


def patient_group_split(patient_ids, fraction: float, seed: int):
    patient_ids = np.asarray(patient_ids).astype(str)
    patients = np.unique(patient_ids)
    if len(patients) < 2:
        raise ValueError("Need at least two patients for segmentation inner split")
    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    n_valid = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
    valid_patients = set(shuffled[:n_valid])
    valid = np.flatnonzero(np.isin(patient_ids, list(valid_patients)))
    train = np.flatnonzero(~np.isin(patient_ids, list(valid_patients)))
    if set(patient_ids[train]) & set(patient_ids[valid]):
        raise AssertionError("Patient leakage in segmentation inner split")
    return train, valid


def patient_inner_split(y, patient_ids, fraction: float, seed: int):
    from sklearn.model_selection import StratifiedShuffleSplit

    y = np.asarray(y).astype(int)
    patient_ids = np.asarray(patient_ids).astype(str)

    p = pd.DataFrame({"patient_id": patient_ids, "y": y}).groupby(
        "patient_id", as_index=False
    ).agg(y=("y", "max"))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=fraction,
        random_state=seed,
    )
    p_tr, p_va = next(splitter.split(p["patient_id"], p["y"]))

    tr_pat = set(p.iloc[p_tr]["patient_id"])
    va_pat = set(p.iloc[p_va]["patient_id"])
    tr = np.flatnonzero(np.isin(patient_ids, list(tr_pat)))
    va = np.flatnonzero(np.isin(patient_ids, list(va_pat)))

    if set(patient_ids[tr]) & set(patient_ids[va]):
        raise AssertionError("Patient leakage in inner split")
    return tr, va


def safe_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def safe_ap(y, p) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    return float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, p))


def brier(y, p) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(np.asarray(y).astype(int), np.asarray(p).astype(float)))
