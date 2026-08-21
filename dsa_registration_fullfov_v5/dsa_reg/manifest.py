from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd

REQUIRED_COLUMNS = [
    "split", "patient_id", "series_uid", "series_id",
    "pre_reference_image_path", "post_reference_image_path",
    "pre_mask_path", "post_mask_path", "pre_frame_paths", "post_frame_paths",
    "pre_n_frames", "post_n_frames",
]

# Canonical internal names <- unified 2026-08-18 master-manifest names.
# The adapter is explicit so a future schema change fails loudly instead of silently
# pairing the wrong image, mask, or phase.
UNIFIED_COLUMN_MAP = {
    "png2d_image_path_pre": "pre_reference_image_path",
    "png2d_image_path_post": "post_reference_image_path",
    "png2d_mask_path_pre": "pre_mask_path",
    "png2d_mask_path_post": "post_mask_path",
    "n_pre_frames": "pre_n_frames",
    "n_post_frames": "post_n_frames",
    "png2d_phase_uid_pre": "pre_phase_uid",
    "png2d_phase_uid_post": "post_phase_uid",
    "png2d_mapping_method_pre": "pre_mapping_method",
    "png2d_mapping_method_post": "post_mapping_method",
    "png2d_identity_pearson_correlation_pre": "pre_mapping_score",
    "png2d_identity_pearson_correlation_post": "post_mapping_score",
}


def normalize_manifest_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return the versioned project schema without mutating the source table."""
    out = df.copy()
    for source, target in UNIFIED_COLUMN_MAP.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = normalize_manifest_schema(pd.read_csv(path))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df


def apply_path_remap(path: str, remap: Dict[str, str]) -> str:
    s = str(path)
    for old, new in sorted(remap.items(), key=lambda kv: len(kv[0]), reverse=True):
        if s.startswith(old):
            return new + s[len(old):]
    return s


def parse_frame_paths(cell: str, remap: Dict[str, str] | None = None) -> List[str]:
    remap = remap or {}
    paths = [p for p in str(cell).split("|") if p]
    return [apply_path_remap(p, remap) for p in paths]


def frame_number(path: str | Path) -> int:
    m = re.search(r"-(\d+)\.[A-Za-z0-9]+$", Path(path).name)
    if not m:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(m.group(1))


def frozen_candidate_numbers(json_text: str, view_name: str = "contrast_core20") -> List[int]:
    try:
        blocks = json.loads(json_text)
    except Exception:
        return []
    out: List[int] = []
    for block in blocks:
        view = block.get("view_indices", {})
        vals = view.get(view_name, block.get("indices", []))
        out.extend(int(v) for v in vals)
    return sorted(set(out))


def select_series_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "all":
        return df.copy()
    if policy == "best_mapping":
        work = df.copy()
        if {"pre_mapping_score", "post_mapping_score"}.issubset(work.columns):
            work["_pair_mapping_score"] = work[["pre_mapping_score", "post_mapping_score"]].min(axis=1)
        else:
            work["_pair_mapping_score"] = 0.0
        work = work.sort_values(["patient_id", "_pair_mapping_score", "series_uid"],
                                ascending=[True, False, True])
        return work.drop_duplicates("patient_id", keep="first").drop(columns="_pair_mapping_score")
    if policy == "error":
        dup = df[df.duplicated("patient_id", keep=False)]
        if len(dup):
            raise ValueError(f"Repeated patient_id exists; examples: {dup.patient_id.unique()[:10].tolist()}")
        return df.copy()
    raise ValueError(f"Unknown patient_series_policy={policy}")


def assert_no_cross_split_patient_leakage(df: pd.DataFrame) -> None:
    nsplit = df.groupby("patient_id")["split"].nunique()
    bad = nsplit[nsplit > 1]
    if len(bad):
        raise ValueError(f"Patient leakage across split: {bad.index.tolist()[:20]}")


def audit_manifest(df: pd.DataFrame, remap: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        pre_frames = parse_frame_paths(r.pre_frame_paths, remap)
        post_frames = parse_frame_paths(r.post_frame_paths, remap)
        paths = {
            "pre_reference": apply_path_remap(r.pre_reference_image_path, remap),
            "post_reference": apply_path_remap(r.post_reference_image_path, remap),
            "pre_mask": apply_path_remap(r.pre_mask_path, remap),
            "post_mask": apply_path_remap(r.post_mask_path, remap),
        }
        rows.append({
            "split": r.split,
            "patient_id": r.patient_id,
            "series_uid": r.series_uid,
            "series_id": r.series_id,
            "pre_count_manifest": int(r.pre_n_frames),
            "post_count_manifest": int(r.post_n_frames),
            "pre_count_paths": len(pre_frames),
            "post_count_paths": len(post_frames),
            "frame_count_ok": len(pre_frames) == int(r.pre_n_frames) and len(post_frames) == int(r.post_n_frames),
            **{f"{k}_exists": Path(v).exists() for k, v in paths.items()},
            "pre_all_frames_exist": all(Path(p).exists() for p in pre_frames),
            "post_all_frames_exist": all(Path(p).exists() for p in post_frames),
        })
    return pd.DataFrame(rows)
