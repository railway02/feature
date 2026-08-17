"""Loading and fail-closed alignment checks for strict main fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {"train_rows": 781, "valid_rows": 207, "spatial_dim": 1024, "temporal_dim": 10240}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def _as_str(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(str)


def load_and_audit(spatial_train: str | Path, spatial_valid: str | Path, cave_train: str | Path, cave_valid: str | Path) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    st, sv, ct, cv = (_read(p) for p in (spatial_train, spatial_valid, cave_train, cave_valid))
    required_st = {"series_uid", "patient_id", "target", "outer_fold", "oof_source_fold", "pred_combined_by_fold", "pred_combined_oof"}
    required_sv = {"series_uid", "patient_id", "target", "pred_combined_by_fold"}
    required_ct = {"series_uid", "patient_id", "target", "fold", "deep"}
    required_cv = {"series_uid", "patient_id", "target", "deep"}
    for name, raw, required in (("spatial_train", st, required_st), ("spatial_valid", sv, required_sv), ("cave_train", ct, required_ct), ("cave_valid", cv, required_cv)):
        missing = required - set(raw)
        if missing:
            raise KeyError(f"{name}: missing {sorted(missing)}")
    for split, s, c, n in (("train", st, ct, EXPECTED["train_rows"]), ("valid", sv, cv, EXPECTED["valid_rows"])):
        uid_s, uid_c = _as_str(s["series_uid"]), _as_str(c["series_uid"])
        pid_s, pid_c = _as_str(s["patient_id"]), _as_str(c["patient_id"])
        if len(uid_s) != n or len(uid_c) != n:
            raise AssertionError(f"{split}: expected {n} rows")
        if not np.array_equal(uid_s, uid_c) or not np.array_equal(pid_s, pid_c):
            raise AssertionError(f"{split}: spatial/CAVE row order mismatch")
        if not np.array_equal(s["target"], c["target"]):
            raise AssertionError(f"{split}: target mismatch")
        if len(np.unique(uid_s)) != n:
            raise AssertionError(f"{split}: duplicate series_uid")
    if not np.array_equal(st["outer_fold"], ct["fold"]):
        raise AssertionError("train: outer-fold mismatch")
    folds = np.asarray(st["outer_fold"], dtype=np.int64)
    if set(np.unique(folds)) != {1, 2, 3, 4, 5}:
        raise AssertionError(f"unexpected folds: {np.unique(folds)}")
    if not np.array_equal(st["oof_source_fold"], folds):
        raise AssertionError("train: OOF 2D source fold differs from outer fold")
    source_view = st["pred_combined_by_fold"][np.arange(len(folds)), folds - 1]
    if not np.array_equal(source_view, st["pred_combined_oof"]):
        raise AssertionError("train: OOF view does not equal by-fold source column")
    if st["pred_combined_by_fold"].shape != (781, 5, 1024) or sv["pred_combined_by_fold"].shape != (207, 5, 1024):
        raise AssertionError("unexpected 2D by-fold shape")
    if ct["deep"].shape != (781, 10240) or cv["deep"].shape != (207, 10240):
        raise AssertionError("unexpected CAVE deep shape")
    for name, x in (("2d_train", st["pred_combined_by_fold"]), ("2d_valid", sv["pred_combined_by_fold"]), ("cave_train", ct["deep"]), ("cave_valid", cv["deep"])):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise AssertionError(f"{name}: must be finite float32")
    train_uid, valid_uid = _as_str(st["series_uid"]), _as_str(sv["series_uid"])
    train_pid, valid_pid = _as_str(st["patient_id"]), _as_str(sv["patient_id"])
    if set(train_uid) & set(valid_uid) or set(train_pid) & set(valid_pid):
        raise AssertionError("Train/Valid identifier leakage")
    audit = {
        "status": "PASS", "expected": EXPECTED,
        "train": {"rows": 781, "uid_order_exact": True, "patient_order_exact": True, "target_exact": True, "fold_exact": True, "uid_unique": True, "finite_2d": True, "finite_cave_deep": True, "fold_counts": {str(k): int((folds == k).sum()) for k in range(1, 6)}},
        "valid": {"rows": 207, "uid_order_exact": True, "patient_order_exact": True, "target_exact": True, "uid_unique": True, "finite_2d": True, "finite_cave_deep": True},
        "cross_split_uid_overlap": 0, "cross_split_patient_overlap": 0,
        "formal_2d_source": "pred_combined_by_fold=[G_pre,soft_PredROI_pre,G_post,soft_PredROI_post]; GTROI/mask and latent averaging prohibited",
    }
    train_df = pd.DataFrame({"row_index": np.arange(781), "series_uid": train_uid, "patient_id": train_pid, "outer_fold": folds, "oof_source_fold": st["oof_source_fold"].astype(int)})
    valid_df = pd.DataFrame({"row_index": np.arange(207), "series_uid": valid_uid, "patient_id": valid_pid, "available_spatial_source_folds": ["1|2|3|4|5"] * 207})
    train = {"series_uid": train_uid, "patient_id": train_pid, "target": st["target"].astype(np.int64), "fold": folds, "spatial_by_fold": st["pred_combined_by_fold"], "spatial_oof": st["pred_combined_oof"], "temporal": ct["deep"]}
    valid = {"series_uid": valid_uid, "patient_id": valid_pid, "target": sv["target"].astype(np.int64), "spatial_by_fold": sv["pred_combined_by_fold"], "temporal": cv["deep"]}
    return train, valid, audit, train_df, valid_df


def write_json(path: str | Path, value: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
