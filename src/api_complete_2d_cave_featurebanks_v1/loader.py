"""Shared immutable-source helpers for the complete 2D--CAVE inventory bank."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


WORKSPACE = Path("/root/autodl-tmp")
ROOT = WORKSPACE / "aneurysm"
V6_ROOT = ROOT / "outputs/api_png2d_spatial_backbones_v6_strict"
CAVE_ROOT = ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/api_complete_2d_cave_featurebanks_v1"
ALL2D_MANIFEST = V6_ROOT / "all2d_segmentation/all2d_segmentation_manifest.csv"
AVAILABILITY = CAVE_ROOT / "tables/local_eligible/local_phase_availability.csv"
CHECKPOINT_ROOT = V6_ROOT / "expanded_strict/segmentation/segresnet"
V6_CONFIG = ROOT / "configs/api_png2d_spatial_backbones_v6_strict.json"
V5_CASE_MANIFEST = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict/case_manifest.csv"
LEGAL_ROOT = V6_ROOT / "expanded_strict"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_npz(path: str | Path, **values: np.ndarray) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}.tmp.npz")
    np.savez_compressed(tmp, **values)
    tmp.replace(p)


def read_all2d_manifest() -> pd.DataFrame:
    x = pd.read_csv(ALL2D_MANIFEST, dtype=str, keep_default_na=False)
    required = {"segmentation_key", "series_uid", "patient_id", "phase", "image_path", "mask_path"}
    if required - set(x): raise KeyError(f"all2d manifest missing {sorted(required-set(x))}")
    x["phase"] = x["phase"].str.lower()
    if len(x) != 2233 or x["segmentation_key"].nunique() != 2233: raise AssertionError("all2d inventory is not 2233 unique phases")
    return x.sort_values("segmentation_key").reset_index(drop=True)


def read_availability() -> pd.DataFrame:
    x = pd.read_csv(AVAILABILITY, dtype=str, keep_default_na=False)
    required = {"phase_uid", "split", "patient_id", "series_uid", "phase", "png_key", "local_eligible", "runtime_feature_excluded", "local_feature_available", "feature_exclusion_reason"}
    if required - set(x): raise KeyError(f"availability missing {sorted(required-set(x))}")
    x["phase"] = x["phase"].str.lower()
    if len(x) != 2622 or x["phase_uid"].nunique() != 2622: raise AssertionError("CAVE availability is not 2622 unique phases")
    return x.sort_values("phase_uid").reset_index(drop=True)


def prepare_image(image_path: str, input_size: int = 768, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Exact frozen image branch; intentionally never opens a GT mask."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise FileNotFoundError(image_path)
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0: raise ValueError(f"non-finite image: {image_path}")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    normalized = np.zeros_like(values, dtype=np.float32) if hi <= lo else np.clip((values-lo)/(hi-lo), 0.0, 1.0).astype(np.float32)
    height, width = normalized.shape
    scale = min(input_size / height, input_size / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(normalized, (new_w, new_h), interpolation=interpolation)
    top, left = (input_size-new_h)//2, (input_size-new_w)//2
    output = cv2.copyMakeBorder(resized, top, input_size-new_h-top, left, input_size-new_w-left, cv2.BORDER_CONSTANT, value=float(np.median(resized)))
    if output.shape != (input_size, input_size) or not np.isfinite(output).all(): raise AssertionError("invalid frozen preprocessing output")
    return output.astype(np.float32)


def outcome_phase_fold_map() -> dict[str, int]:
    x = pd.read_csv(V5_CASE_MANIFEST, dtype={"patient_id": str, "series_uid": str})
    required = {"split", "fold", "pre_png_key", "post_png_key"}
    if required - set(x): raise KeyError(f"case manifest missing {sorted(required-set(x))}")
    train = x.loc[x["split"].eq("Train")]
    result: dict[str, int] = {}
    for row in train.itertuples(index=False):
        for key in (str(row.pre_png_key), str(row.post_png_key)):
            if key in result and result[key] != int(row.fold): raise AssertionError(f"conflicting outcome fold for {key}")
            result[key] = int(row.fold)
    return result


def checkpoint_seen_patients() -> tuple[dict[int, set[str]], dict[int, str]]:
    """Read actual legal segmentation manifests; inner-valid is also treated seen."""
    seen, hashes = {}, {}
    for fold in range(1, 6):
        p = LEGAL_ROOT / f"fold_{fold}/segmentation_legal_split_manifest.csv"
        x = pd.read_csv(p, dtype={"patient_id": str})
        if "patient_id" not in x: raise KeyError(f"{p}: patient_id missing")
        seen[fold] = set(x["patient_id"].astype(str))
        hashes[fold] = sha256(p)
    return seen, hashes


def no_label_fields(fields: set[str], context: str) -> None:
    forbidden = fields & {"target", "label", "outcome", "y", "adverse_outcome"}
    if forbidden: raise AssertionError(f"{context}: forbidden label fields {sorted(forbidden)}")
