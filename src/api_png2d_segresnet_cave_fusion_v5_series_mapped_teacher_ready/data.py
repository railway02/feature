from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import load_temporal, load_train_folds, normalize_id, resolve_path


def build_case_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    mapping_path = resolve_path(
        cfg["data"]["phase_mapping_manifest"],
        cfg["project_root"],
    )
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    required = {
        "phase_uid", "split", "patient_id", "series_uid", "phase",
        "png_key", "reference_image_path", "mask_path",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise KeyError(f"{mapping_path}: missing mapping columns {sorted(missing)}")

    mapping = mapping.copy()
    mapping["patient_id"] = mapping["patient_id"].map(normalize_id)
    mapping["series_uid"] = mapping["series_uid"].astype(str)
    mapping["phase"] = mapping["phase"].str.strip().str.casefold()
    expected_phases = {str(x).casefold() for x in cfg["data"]["phases"]}
    if expected_phases != {"pre", "post"}:
        raise ValueError(f"data.phases must be Pre/Post, got {sorted(expected_phases)}")
    bad_phase = mapping[~mapping["phase"].isin(expected_phases)]
    if len(bad_phase):
        raise RuntimeError(
            f"Mapping contains unsupported phases: {sorted(bad_phase['phase'].unique())}"
        )
    duplicate = mapping.duplicated(["series_uid", "phase"], keep=False)
    if duplicate.any():
        examples = mapping.loc[
            duplicate, ["series_uid", "phase", "phase_uid"]
        ].head(10)
        raise RuntimeError(
            f"Duplicate series_uid+phase mappings:\n{examples.to_string(index=False)}"
        )

    lookup = mapping.set_index(["series_uid", "phase"], verify_integrity=True)
    rows = []
    for split in ("Train", "Valid"):
        raw = load_temporal(cfg, split)
        folds = (
            load_train_folds(cfg, raw)
            if split == "Train"
            else np.zeros(len(raw["target"]), dtype=int)
        )

        for i in range(len(raw["target"])):
            patient = normalize_id(raw["patient_id"][i])
            row = {
                "split": split,
                "task_row": int(i),
                "series_uid": str(raw["series_uid"][i]),
                "patient_id": patient,
                "target": int(raw["target"][i]),
                "fold": int(folds[i]),
            }
            for phase in cfg["data"]["phases"]:
                key = str(phase).casefold()
                map_key = (row["series_uid"], key)
                if map_key not in lookup.index:
                    raise KeyError(
                        f"Missing mapping for series_uid={row['series_uid']} phase={key}"
                    )
                mapped = lookup.loc[map_key]
                mapped_patient = normalize_id(mapped["patient_id"])
                mapped_split = str(mapped["split"])
                if mapped_patient != patient:
                    raise RuntimeError(
                        f"Patient mismatch for {map_key}: "
                        f"task={patient}, mapping={mapped_patient}"
                    )
                if mapped_split != split:
                    raise RuntimeError(
                        f"Split mismatch for {map_key}: "
                        f"task={split}, mapping={mapped_split}"
                    )
                row[f"{key}_phase_uid"] = str(mapped["phase_uid"])
                row[f"{key}_png_key"] = str(mapped["png_key"])
                row[f"{key}_image"] = str(
                    Path(mapped["reference_image_path"]).resolve()
                )
                row[f"{key}_mask"] = str(Path(mapped["mask_path"]).resolve())
            rows.append(row)

    return pd.DataFrame(rows)


def build_segmentation_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    inventory_path = resolve_path(
        cfg["data"]["segmentation_inventory"],
        cfg["project_root"],
    )
    frame = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    required = {
        "png_key", "image_path", "mask_path", "image_shape",
        "mask_shape", "mask_nonzero_pixels",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{inventory_path}: missing inventory columns {sorted(missing)}")
    if frame["png_key"].duplicated().any():
        raise RuntimeError("Segmentation inventory has duplicate png_key")

    phase = frame["png_key"].str.extract(r"_(Pre|Post)$", expand=False)
    patient = frame["png_key"].str.extract(r"^(\d+)", expand=False)
    if phase.isna().any() or patient.isna().any():
        bad = frame.loc[phase.isna() | patient.isna(), "png_key"].head(10).tolist()
        raise RuntimeError(f"Cannot parse patient/phase from png_key: {bad}")

    out = pd.DataFrame({
        "segmentation_key": frame["png_key"].astype(str),
        "patient_id": patient.map(normalize_id),
        "phase": phase.astype(str),
        "image_path": frame["image_path"].map(lambda x: str(Path(x).resolve())),
        "mask_path": frame["mask_path"].map(lambda x: str(Path(x).resolve())),
        "image_shape": frame["image_shape"].astype(str),
        "mask_shape": frame["mask_shape"].astype(str),
        "mask_nonzero_pixels": pd.to_numeric(
            frame["mask_nonzero_pixels"], errors="raise"
        ).astype(int),
    })
    return out.sort_values("segmentation_key").reset_index(drop=True)


def inspect_segmentation_inventory(manifest: pd.DataFrame):
    rows, failures = [], []
    for r in manifest.itertuples(index=False):
        ip = Path(r.image_path)
        mp = Path(r.mask_path)
        base = {
            "segmentation_key": r.segmentation_key,
            "patient_id": r.patient_id,
            "phase": r.phase,
            "image_path": str(ip),
            "mask_path": str(mp),
        }
        errors = []
        if not ip.is_file():
            errors.append(f"missing image:{ip}")
        if not mp.is_file():
            errors.append(f"missing mask:{mp}")
        if ip.name != mp.name:
            errors.append(f"basename mismatch:{ip.name}!={mp.name}")
        if str(r.image_shape) != str(r.mask_shape):
            errors.append(f"shape mismatch:{r.image_shape}!={r.mask_shape}")
        if int(r.mask_nonzero_pixels) <= 0:
            errors.append("empty mask")
        if errors:
            failures.append({**base, "status": "fail", "error": "|".join(errors)})
        else:
            rows.append({
                **base,
                "image_shape": str(r.image_shape),
                "mask_shape": str(r.mask_shape),
                "mask_pixels": int(r.mask_nonzero_pixels),
                "status": "ok",
            })
    return pd.DataFrame(rows), pd.DataFrame(failures)


def inspect_files(manifest: pd.DataFrame, cfg: dict[str, Any]):
    rows, failures = [], []
    for r in manifest.itertuples(index=False):
        for phase in cfg["data"]["phases"]:
            key = phase.lower()
            ip = Path(getattr(r, f"{key}_image"))
            mp = Path(getattr(r, f"{key}_mask"))
            base = {
                "split": r.split,
                "series_uid": r.series_uid,
                "patient_id": r.patient_id,
                "phase": phase,
                "image_path": str(ip),
                "mask_path": str(mp),
            }
            try:
                if not ip.is_file():
                    raise FileNotFoundError(f"image:{ip}")
                if not mp.is_file():
                    raise FileNotFoundError(f"mask:{mp}")
                if ip.name != mp.name:
                    raise ValueError(f"basename mismatch {ip.name} vs {mp.name}")

                image = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError("image unreadable")
                if mask is None:
                    raise ValueError("mask unreadable")
                if image.shape != mask.shape:
                    raise ValueError(f"shape mismatch {image.shape} vs {mask.shape}")

                fg = mask > 0
                if not fg.any():
                    raise ValueError("empty mask")

                ys, xs = np.where(fg)
                rows.append({
                    **base,
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    "mask_pixels": int(fg.sum()),
                    "mask_area_ratio": float(fg.mean()),
                    "bbox_x0": int(xs.min()),
                    "bbox_y0": int(ys.min()),
                    "bbox_x1": int(xs.max()) + 1,
                    "bbox_y1": int(ys.max()) + 1,
                    "status": "ok",
                })
            except Exception as e:
                failures.append({
                    **base,
                    "status": "fail",
                    "error": f"{type(e).__name__}:{e}",
                })
    return pd.DataFrame(rows), pd.DataFrame(failures)


def percentile_normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("No finite pixels")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def letterbox_pair(image: np.ndarray, mask: np.ndarray, target: int):
    h, w = image.shape
    scale = min(target / h, target / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    ir = cv2.resize(image, (nw, nh), interpolation=interp)
    mr = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)

    top = (target - nh) // 2
    bottom = target - nh - top
    left = (target - nw) // 2
    right = target - nw - left

    fill = float(np.median(ir))
    ir = cv2.copyMakeBorder(ir, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    mr = cv2.copyMakeBorder(mr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return ir.astype(np.float32), (mr > 0).astype(np.float32)


def prepare_pair(image_path: str, mask_path: str, cfg: dict[str, Any]):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Cannot read {image_path} / {mask_path}")
    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch {image.shape} vs {mask.shape}")

    image = percentile_normalize(
        image,
        float(cfg["spatial"]["percentile_low"]),
        float(cfg["spatial"]["percentile_high"]),
    )
    mask = (mask > 0).astype(np.float32)

    size = int(cfg["spatial"]["input_size"])
    if bool(cfg["spatial"].get("letterbox", True)):
        image, mask = letterbox_pair(image, mask, size)
    else:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    if not (mask > 0).any():
        raise ValueError(f"Mask vanished after resize: {mask_path}")

    return image.astype(np.float32), mask.astype(np.float32)


class SegPhaseDataset(Dataset):
    def __init__(self, case_frame: pd.DataFrame, cfg: dict[str, Any], augment: bool):
        self.cfg = cfg
        self.augment = augment
        rows = []
        if {"image_path", "mask_path", "patient_id"} <= set(case_frame.columns):
            self.rows = case_frame.reset_index(drop=True).copy()
            return
        for r in case_frame.itertuples(index=False):
            for phase in cfg["data"]["phases"]:
                key = phase.lower()
                rows.append({
                    "patient_id": str(r.patient_id),
                    "series_uid": str(r.series_uid),
                    "phase": phase,
                    "image_path": getattr(r, f"{key}_image"),
                    "mask_path": getattr(r, f"{key}_mask"),
                })
        self.rows = pd.DataFrame(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows.iloc[index]
        image, mask = prepare_pair(r["image_path"], r["mask_path"], self.cfg)

        aug = self.cfg["segresnet"].get("augmentation", {})
        if self.augment and bool(aug.get("enabled", True)):
            c = float(aug.get("contrast", 0))
            b = float(aug.get("brightness", 0))
            n = float(aug.get("noise_std", 0))

            if c > 0:
                factor = np.random.uniform(1-c, 1+c)
                mean = float(image.mean())
                image = (image - mean) * factor + mean
            if b > 0:
                image = image + np.random.uniform(-b, b)
            if n > 0:
                image = image + np.random.normal(0, n, image.shape).astype(np.float32)

            image = np.clip(image, 0, 1)

        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            int(index),
        )


class FeaturePhaseDataset(Dataset):
    def __init__(self, case_frame: pd.DataFrame, cfg: dict[str, Any]):
        self.cfg = cfg
        rows = []
        for r in case_frame.itertuples(index=False):
            for phase in cfg["data"]["phases"]:
                key = phase.lower()
                rows.append({
                    "case_row": int(r.task_row),
                    "patient_id": str(r.patient_id),
                    "series_uid": str(r.series_uid),
                    "phase": phase,
                    "image_path": getattr(r, f"{key}_image"),
                    "mask_path": getattr(r, f"{key}_mask"),
                })
        self.rows = pd.DataFrame(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows.iloc[index]
        image, mask = prepare_pair(r["image_path"], r["mask_path"], self.cfg)
        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            int(index),
        )


def estimate_bce_pos_weight(case_frame: pd.DataFrame, cfg: dict[str, Any]) -> float:
    fg = 0
    total = 0
    if "mask_path" in case_frame.columns:
        mask_paths = case_frame["mask_path"].astype(str).tolist()
    else:
        mask_paths = [
            getattr(r, f"{phase.lower()}_mask")
            for r in case_frame.itertuples(index=False)
            for phase in cfg["data"]["phases"]
        ]
    for mp in mask_paths:
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mp)
        binary = mask > 0
        fg += int(binary.sum())
        total += int(binary.size)

    bg = max(1, total - fg)
    fg = max(1, fg)

    # sqrt ratio is deliberately milder because Dice already handles imbalance.
    raw = float(np.sqrt(bg / fg))
    return float(np.clip(raw, 1.0, float(cfg["segresnet"]["max_bce_pos_weight"])))
