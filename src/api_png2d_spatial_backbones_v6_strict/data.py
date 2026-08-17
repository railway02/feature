from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augmentations import augment_pair
from common import canonical_hash


def load_train_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(
        cfg["sources"]["v5_case_manifest"],
        dtype={"patient_id": str, "series_uid": str},
    )
    train = frame.loc[frame["split"].eq("Train")].copy().reset_index(drop=True)
    train["fold"] = pd.to_numeric(train["fold"], errors="raise").astype(int)
    train["target"] = pd.to_numeric(train["target"], errors="raise").astype(int)
    expected = int(cfg["data"]["expected_train_series"])
    if len(train) != expected or train["series_uid"].nunique() != expected:
        raise RuntimeError(f"Unexpected Train manifest rows/UIDs: {len(train)}/{train['series_uid'].nunique()}")
    patient_folds = train.groupby("patient_id")["fold"].nunique()
    if int(patient_folds.max()) != 1:
        raise RuntimeError("A Train patient appears in more than one outer fold")
    return train


def development_split(train: pd.DataFrame, outer_fold: int, cfg: dict[str, Any]):
    development = train.loc[train["fold"].ne(int(outer_fold))].copy()
    holdout = train.loc[train["fold"].eq(int(outer_fold))].copy()
    if set(development["patient_id"]) & set(holdout["patient_id"]):
        raise RuntimeError("Outer patient leakage")
    # Match the frozen v5 common.patient_group_split implementation exactly:
    # it calls np.unique before shuffling, so patient order is sorted rather
    # than first-appearance order from the case manifest.
    patient_ids = np.unique(development["patient_id"].astype(str).to_numpy())
    rng = np.random.default_rng(int(cfg["development"]["base_seed"]) + int(outer_fold) * 100)
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)
    fraction = float(cfg["development"]["inner_val_fraction"])
    n_valid = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
    valid_patients = set(shuffled[:n_valid])
    inner_valid = development.loc[development["patient_id"].isin(valid_patients)].copy()
    inner_train = development.loc[~development["patient_id"].isin(valid_patients)].copy()
    if set(inner_train["patient_id"]) & set(inner_valid["patient_id"]):
        raise RuntimeError("Inner patient leakage")
    return inner_train, inner_valid, holdout


def split_manifest(train: pd.DataFrame, outer_fold: int, cfg: dict[str, Any]) -> pd.DataFrame:
    inner_train, inner_valid, holdout = development_split(train, outer_fold, cfg)
    return pd.concat([
        inner_train.assign(development_partition="inner_train"),
        inner_valid.assign(development_partition="inner_valid"),
        holdout.assign(development_partition="forbidden_outer_holdout_not_evaluated"),
    ], ignore_index=True)


def split_hash(frame: pd.DataFrame) -> str:
    columns = ["series_uid", "patient_id", "fold", "development_partition"]
    frame = frame.copy()
    if "fold" not in frame:
        frame["fold"] = 0
    records = frame[columns].astype(str).sort_values(columns).to_dict("records")
    return canonical_hash(records)


def unroll_phase_rows(case_frame: pd.DataFrame, phases: list[str]) -> pd.DataFrame:
    if {"image_path", "mask_path", "patient_id", "phase"} <= set(case_frame.columns):
        direct = case_frame.copy().reset_index(drop=True)
        if "series_uid" not in direct:
            direct["series_uid"] = direct.get("segmentation_key", direct.index.astype(str)).astype(str)
        if "fold" not in direct:
            direct["fold"] = 0
        return direct[["patient_id", "series_uid", "fold", "phase", "image_path", "mask_path"]].copy()
    rows = []
    for row in case_frame.itertuples(index=False):
        for phase in phases:
            key = phase.lower()
            rows.append({
                "patient_id": str(row.patient_id),
                "series_uid": str(row.series_uid),
                "outer_fold": int(row.fold),
                "phase": phase,
                "image_path": str(getattr(row, f"{key}_image")),
                "mask_path": str(getattr(row, f"{key}_mask")),
            })
    return pd.DataFrame(rows)


def load_all2d_segmentation_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    inventory = pd.read_csv(cfg["sources"]["segmentation_inventory"], dtype=str, keep_default_na=False)
    required = {"png_key", "image_path", "mask_path", "image_shape", "mask_shape", "mask_nonzero_pixels"}
    missing = required - set(inventory.columns)
    if missing:
        raise KeyError(f"Segmentation inventory is missing columns: {sorted(missing)}")
    if inventory["png_key"].duplicated().any():
        raise RuntimeError("Segmentation inventory contains duplicate png_key")
    phase = inventory["png_key"].str.extract(r"_(Pre|Post)$", expand=False)
    patient = inventory["png_key"].str.extract(r"^(\d+)", expand=False)
    if phase.isna().any() or patient.isna().any():
        raise RuntimeError("Cannot parse patient/phase from one or more png_key values")
    manifest = pd.DataFrame({
        "segmentation_key": inventory["png_key"].astype(str),
        "series_uid": inventory["png_key"].astype(str),
        "patient_id": patient.astype(str),
        "phase": phase.astype(str),
        "image_path": inventory["image_path"].map(lambda value: str(Path(value).resolve())),
        "mask_path": inventory["mask_path"].map(lambda value: str(Path(value).resolve())),
        "image_shape": inventory["image_shape"].astype(str),
        "mask_shape": inventory["mask_shape"].astype(str),
        "mask_nonzero_pixels": pd.to_numeric(inventory["mask_nonzero_pixels"], errors="raise").astype(int),
    }).sort_values("segmentation_key").reset_index(drop=True)
    expected = cfg["all2d_segmentation"]
    if len(manifest) != int(expected["expected_phase_rows"]):
        raise RuntimeError(f"Expected {expected['expected_phase_rows']} all-2D rows, found {len(manifest)}")
    if manifest["patient_id"].nunique() != int(expected["expected_patients"]):
        raise RuntimeError(f"Expected {expected['expected_patients']} all-2D patients, found {manifest['patient_id'].nunique()}")
    return manifest


def all2d_inner_split(manifest: pd.DataFrame, cfg: dict[str, Any]):
    patient_ids = np.unique(manifest["patient_id"].astype(str).to_numpy())
    rng = np.random.default_rng(int(cfg["all2d_segmentation"]["seed"]))
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)
    fraction = float(cfg["all2d_segmentation"]["inner_val_fraction"])
    n_valid = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
    valid_patients = set(shuffled[:n_valid])
    inner_valid = manifest.loc[manifest["patient_id"].isin(valid_patients)].copy()
    inner_train = manifest.loc[~manifest["patient_id"].isin(valid_patients)].copy()
    if set(inner_train["patient_id"]) & set(inner_valid["patient_id"]):
        raise RuntimeError("All-2D inner split has patient leakage")
    split = pd.concat([
        inner_train.assign(development_partition="inner_train"),
        inner_valid.assign(development_partition="inner_valid"),
    ], ignore_index=True)
    return inner_train, inner_valid, split


def _metadata_patient_ids(path: str, column: str) -> set[str]:
    metadata = pd.read_excel(path)
    if column not in metadata:
        raise KeyError(f"{path}: missing metadata patient column {column}")
    values = pd.to_numeric(metadata[column], errors="coerce").dropna().astype(int).astype(str)
    return set(values)


def build_expanded_strict_population(cfg: dict[str, Any]):
    """Build the user-specified 1,780/453 segmentation pools from metadata.

    Returns train_pool, valid_pool, adverse_train_case_manifest, audit.  The
    two manual Train rows are validated as the exact difference between the
    metadata-derived Train pool and the phase-eligible Train manifest.
    """
    spec = cfg["expanded_strict_segmentation"]
    inventory = pd.read_csv(cfg["sources"]["segmentation_inventory"], dtype=str, keep_default_na=False)
    train_patients = _metadata_patient_ids(spec["segmentation_train_metadata"], spec["metadata_patient_column"])
    valid_patients = _metadata_patient_ids(spec["segmentation_valid_metadata"], spec["metadata_patient_column"])
    if train_patients & valid_patients:
        raise RuntimeError("Train.xlsx and valid.xlsx patients overlap")
    inventory = inventory.copy()
    inventory["patient_id"] = inventory["png_key"].str.extract(r"^(\d+)", expand=False)
    inventory["phase"] = inventory["png_key"].str.extract(r"_(Pre|Post)$", expand=False)
    if inventory["patient_id"].isna().any() or inventory["phase"].isna().any():
        raise RuntimeError("Inventory has unparseable patient_id or phase")
    train_pool = inventory.loc[inventory["patient_id"].isin(train_patients)].copy()
    valid_pool = inventory.loc[inventory["patient_id"].isin(valid_patients)].copy()
    unassigned = inventory.loc[~inventory["patient_id"].isin(train_patients | valid_patients)]
    if len(train_pool) != int(spec["expected_train_rows"]) or len(valid_pool) != int(spec["expected_valid_rows"]) or len(unassigned):
        raise RuntimeError(f"Metadata-derived pools invalid: train={len(train_pool)}, valid={len(valid_pool)}, unassigned={len(unassigned)}")
    for frame, name in ((train_pool, "Train"), (valid_pool, "Valid")):
        frame["series_uid"] = frame["png_key"].astype(str)
        frame["segmentation_key"] = frame["png_key"].astype(str)
        frame["image_path"] = frame["image_path"].map(lambda value: str(Path(value).resolve()))
        frame["mask_path"] = frame["mask_path"].map(lambda value: str(Path(value).resolve()))
        frame["segmentation_split"] = name
        frame["fold"] = 0

    eligible = pd.read_csv(spec["phase_eligible_manifest"], dtype=str, keep_default_na=False)
    eligible_train = set(eligible.loc[eligible["split"].eq("Train"), "png_key"].astype(str))
    actual_train_keys = set(train_pool["png_key"].astype(str))
    manual = actual_train_keys - eligible_train
    if manual != set(spec["manual_train_png_keys"]):
        raise RuntimeError(f"Manual Train PNG difference mismatch: {sorted(manual)}")
    if len(eligible_train & actual_train_keys) != 1778:
        raise RuntimeError("Expected 1,778 phase-eligible Train PNG rows")

    adverse = load_train_manifest(cfg)
    adverse_png_to_fold = {}
    adverse_patient_to_fold = {}
    adverse_patients = set(adverse["patient_id"].astype(str))
    for row in adverse.itertuples(index=False):
        adverse_png_to_fold[str(row.pre_png_key)] = int(row.fold)
        adverse_png_to_fold[str(row.post_png_key)] = int(row.fold)
        adverse_patient_to_fold[str(row.patient_id)] = int(row.fold)
    # Every extra series of an adverse Train patient inherits that patient's
    # outer fold even when its PNG key is absent from the outcome cohort.
    train_pool["fold"] = train_pool["patient_id"].map(adverse_patient_to_fold).fillna(0).astype(int)
    adverse_rows = train_pool.loc[train_pool["fold"].gt(0)].copy()
    if len(adverse_rows) != 1562 + len(spec["adverse_extra_png_keys"]):
        raise RuntimeError(f"Unexpected adverse-patient segmentation rows: {len(adverse_rows)}")
    outcome_png = set(adverse_png_to_fold)
    extras = set(adverse_rows["png_key"]) - outcome_png
    if extras != set(spec["adverse_extra_png_keys"]):
        raise RuntimeError(f"Adverse extra PNG mismatch: {sorted(extras)}")
    segmentation_only = train_pool.loc[train_pool["fold"].eq(0)].copy()
    if len(segmentation_only) != int(spec["expected_segmentation_only_train_rows"]) or segmentation_only["patient_id"].nunique() != int(spec["expected_segmentation_only_train_patients"]):
        raise RuntimeError("Segmentation-only Train population count mismatch")
    audit = {
        "train_rows": int(len(train_pool)), "valid_rows": int(len(valid_pool)),
        "train_patients": int(train_pool.patient_id.nunique()), "valid_patients": int(valid_pool.patient_id.nunique()),
        "manual_train_png_keys": sorted(manual), "phase_eligible_train_rows": int(len(eligible_train & actual_train_keys)),
        "adverse_outcome_phase_rows": 1562, "adverse_extra_rows": int(len(extras)), "adverse_extra_png_keys": sorted(extras),
        "segmentation_only_rows": int(len(segmentation_only)), "segmentation_only_patients": int(segmentation_only.patient_id.nunique()),
    }
    return train_pool.reset_index(drop=True), valid_pool.reset_index(drop=True), adverse, audit


def expanded_strict_fold_split(cfg: dict[str, Any], outer_fold: int):
    """Construct legal rows and the required adverse-only inner-valid split."""
    pool, valid_pool, adverse, audit = build_expanded_strict_population(cfg)
    spec = cfg["expanded_strict_segmentation"]
    legal = pool.loc[pool["fold"].ne(int(outer_fold))].copy()
    holdout_patients = set(adverse.loc[adverse["fold"].eq(int(outer_fold)), "patient_id"].astype(str))
    if set(legal["patient_id"]) & holdout_patients:
        raise RuntimeError(f"Fold {outer_fold}: holdout patient remained in legal segmentation rows")
    expected = spec["expected_legal_by_outer_fold"][str(outer_fold)]
    if len(legal) != int(expected["rows"]) or legal["patient_id"].nunique() != int(expected["patients"]):
        raise RuntimeError(f"Fold {outer_fold}: legal counts {len(legal)}/{legal.patient_id.nunique()} != {expected}")
    adverse_development = adverse.loc[adverse["fold"].ne(int(outer_fold))].copy()
    patients = np.unique(adverse_development["patient_id"].astype(str).to_numpy())
    seed = int(cfg["development"]["base_seed"]) + int(outer_fold) * 100
    rng = np.random.default_rng(seed); shuffled = patients.copy(); rng.shuffle(shuffled)
    n_valid = min(
        len(shuffled) - 1,
        max(1, int(round(len(shuffled) * float(cfg["development"]["inner_val_fraction"])))),
    )
    inner_valid_patients = set(shuffled[:n_valid])
    adverse_development_patients = set(patients)
    legal["development_partition"] = "inner_train"
    legal.loc[legal["patient_id"].isin(inner_valid_patients), "development_partition"] = "inner_valid"
    # All 194 segmentation-only patients are explicitly inner-train; they
    # cannot enter epoch selection.
    legal.loc[~legal["patient_id"].isin(adverse_development_patients), "development_partition"] = spec["segmentation_only_inner_partition"]
    inner_train = legal.loc[legal["development_partition"].eq("inner_train")].copy()
    inner_valid = legal.loc[legal["development_partition"].eq("inner_valid")].copy()
    if not set(inner_valid["patient_id"]).issubset(adverse_development_patients):
        raise RuntimeError("Inner-valid contains segmentation-only patient")
    if set(inner_train["patient_id"]) & set(inner_valid["patient_id"]):
        raise RuntimeError("Inner patient leakage")
    fold_audit = {
        **audit, "outer_fold": int(outer_fold), "legal_rows": int(len(legal)), "legal_patients": int(legal.patient_id.nunique()),
        "inner_train_rows": int(len(inner_train)), "inner_train_patients": int(inner_train.patient_id.nunique()),
        "inner_valid_rows": int(len(inner_valid)), "inner_valid_patients": int(inner_valid.patient_id.nunique()),
        "valid_pool_rows_forbidden": int(len(valid_pool)), "outer_holdout_patients": int(len(holdout_patients)),
    }
    return legal, inner_train, inner_valid, valid_pool, fold_audit


def percentile_normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def letterbox_pair(image: np.ndarray, mask: np.ndarray, target: int):
    height, width = image.shape
    scale = min(target / height, target / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    image_interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=image_interpolation)
    mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left
    image_out = cv2.copyMakeBorder(
        image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=float(np.median(image_resized)),
    )
    mask_out = cv2.copyMakeBorder(mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    audit = {"scale": float(scale), "padding": [int(top), int(bottom), int(left), int(right)]}
    return image_out.astype(np.float32), (mask_out > 0).astype(np.float32), audit


def prepare_pair(image_path: str, mask_path: str, cfg: dict[str, Any]):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask_raw is None:
        raise FileNotFoundError(f"Cannot read image/mask pair: {image_path} / {mask_path}")
    if image.shape != mask_raw.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} != {mask_raw.shape}")
    image = percentile_normalize(image, float(cfg["data"]["percentile_low"]), float(cfg["data"]["percentile_high"]))
    mask = (mask_raw > 0).astype(np.float32)
    if not mask.any():
        raise ValueError(f"Empty mask before resize: {mask_path}")
    image, mask, audit = letterbox_pair(image, mask, int(cfg["data"]["input_size"]))
    if not mask.any():
        raise ValueError(f"Mask vanished after resize: {mask_path}")
    return image, mask, audit


class SegmentationDataset(Dataset):
    def __init__(self, case_frame: pd.DataFrame, cfg: dict[str, Any], augment: bool):
        self.cfg = cfg
        self.augment = bool(augment)
        self.rows = unroll_phase_rows(case_frame, list(cfg["data"]["phases"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[int(index)]
        image, mask, _ = prepare_pair(row.image_path, row.mask_path, self.cfg)
        geometry_applied = False
        geometry_fallback = False
        if self.augment and bool(self.cfg["augmentation"]["enabled"]):
            image, mask, geometry_applied, geometry_fallback = augment_pair(image, mask, self.cfg["augmentation"])
        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            int(index),
            int(geometry_applied),
            int(geometry_fallback),
        )


def estimate_pos_weight(case_frame: pd.DataFrame, cfg: dict[str, Any]) -> float:
    phase_rows = unroll_phase_rows(case_frame, list(cfg["data"]["phases"]))
    foreground = 0
    total = 0
    for path in phase_rows["mask_path"].astype(str):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        binary = mask > 0
        foreground += int(binary.sum())
        total += int(binary.size)
    background = max(1, total - foreground)
    foreground = max(1, foreground)
    lower, upper = [float(x) for x in cfg["loss"]["pos_weight_clip"]]
    return float(np.clip(np.sqrt(background / foreground), lower, upper))
