"""Frozen input contract for the Local Reference Jacobian + HEMO run.

This module deliberately joins identities only by ``split + patient_id +
series_uid``.  It never discovers frames by globbing and it never reads labels.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_json, sha256_file
from .preprocessing_adapter import PairRecord, PhaseRecord, load_local_reference_pairs


G0_REQUIRED = {
    "series_uid", "patient_id", "split", "pre_peak_index", "post_peak_index",
    "pre_peak_score", "post_peak_score", "registration_valid", "linear_valid",
    "nonrigid_valid",
}
TERMINAL_G0_FILES = (
    "rigid.json", "rigid_maps.npz", "rigid_sheet.png", "rigid_syn_0GenericAffine.mat",
    "rigid_syn_1Warp.nii.gz", "rigid_syn_1InverseWarp.nii.gz",
)


@dataclass(frozen=True)
class FrozenPhaseContract:
    record: PhaseRecord
    frozen_peak_index: int
    frozen_peak_score: float
    selected_block_positions: tuple[int, ...]
    frozen_source_indices: tuple[int, ...]

    @property
    def phase_uid(self) -> str:
        return self.record.phase_uid

    @property
    def phase(self) -> str:
        return self.record.phase

    @property
    def frame_paths(self) -> tuple[str, ...]:
        return self.record.frame_paths

    @property
    def expanded_bbox(self):
        return self.record.expanded_bbox

    @property
    def canvas_shape_yx(self) -> tuple[int, int]:
        return self.record.canvas_shape_yx

    @property
    def mask_path(self) -> Path:
        return self.record.mask_path


@dataclass(frozen=True)
class FrozenSeriesContract:
    series_uid: str
    patient_id: str
    split: str
    series_id: str
    pre: FrozenPhaseContract
    post: FrozenPhaseContract
    g0_case_dir: Path
    g0_row: dict[str, Any]


def _norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _load_g0_table(cfg: dict[str, Any], split: str) -> pd.DataFrame:
    root = Path(cfg["paths"]["g0_registration_root"])
    name = "train_registration_qc.csv" if split == "Train" else "valid_registration_qc.csv"
    table = pd.read_csv(root / name, dtype=str, keep_default_na=False)
    missing = sorted(G0_REQUIRED - set(table.columns))
    if missing:
        raise KeyError(f"G0 {split} table missing columns: {missing}")
    if len(table) != int(cfg["expected"]["paired_series_train" if split == "Train" else "paired_series_valid"]):
        raise AssertionError(f"G0 {split} table coverage changed: {len(table)}")
    if table["series_uid"].duplicated().any():
        raise AssertionError(f"G0 {split} table has duplicate series_uid")
    return table


def _source_index(path: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot recover source frame index from frozen path: {path}")
    return int(match.group(1))


def _resolve_block(raw_json: str, phase: PhaseRecord, peak_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve the only acquisition block allowed for interpolation/TDC.

    Pair manifests preserve block *source frame numbers*, whereas the G0 peak is a
    zero-based location in the frozen frame list.  The conversion is explicit and
    rejects ambiguity rather than bridging a gap.
    """
    if not (0 <= peak_index < len(phase.frame_paths)):
        raise ValueError(f"{phase.phase_uid}: frozen peak {peak_index} outside frame list")
    source = tuple(_source_index(path) for path in phase.frame_paths)
    try:
        blocks = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{phase.phase_uid}: invalid frozen temporal blocks JSON") from exc
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{phase.phase_uid}: no frozen temporal blocks")
    peak_source = source[peak_index]
    chosen: list[int] | None = None
    for block in blocks:
        indices = [int(x) for x in block.get("indices", [])]
        if peak_source in indices:
            if chosen is not None:
                raise ValueError(f"{phase.phase_uid}: peak belongs to multiple acquisition blocks")
            chosen = indices
    if chosen is None:
        raise ValueError(f"{phase.phase_uid}: peak source {peak_source} absent from frozen blocks")
    positions = tuple(i for i, value in enumerate(source) if value in set(chosen))
    if not positions or peak_index not in positions:
        raise ValueError(f"{phase.phase_uid}: resolved block does not contain frozen peak")
    selected_sources = tuple(source[i] for i in positions)
    if selected_sources != tuple(chosen):
        raise ValueError(f"{phase.phase_uid}: block/list source order mismatch")
    if tuple(range(positions[0], positions[-1] + 1)) != positions:
        raise ValueError(f"{phase.phase_uid}: selected acquisition block is not contiguous in manifest order")
    return positions, selected_sources


def _pair_manifest_rows(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    path = Path(cfg["paths"]["temporal_pairs"])
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    if rows["series_uid"].duplicated().any():
        raise AssertionError("pair manifest duplicate series_uid")
    return {str(row["series_uid"]): row for row in rows.to_dict("records")}


def _number(row: dict[str, Any], name: str, uid: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{uid}: invalid G0 {name}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{uid}: non-finite G0 {name}")
    return value


def _validate_g0_case(pair: PairRecord, row: dict[str, Any], case_dir: Path) -> None:
    uid = pair.series_uid
    if str(row["series_uid"]) != uid or str(row["split"]) != pair.split:
        raise AssertionError(f"{uid}: G0 identity mismatch")
    if _norm_id(row["patient_id"]) != _norm_id(pair.patient_id):
        raise AssertionError(f"{uid}: G0 patient_id mismatch")
    missing = [name for name in TERMINAL_G0_FILES if not (case_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{uid}: G0 terminal assets missing: {missing}")
    with np.load(case_dir / "rigid_maps.npz", allow_pickle=False) as maps:
        if not {"logj", "disp", "valid", "folding"}.issubset(maps.files):
            raise ValueError(f"{uid}: malformed G0 rigid_maps.npz")
        expected_shape = (pair.post.expanded_bbox.height, pair.post.expanded_bbox.width)
        if maps["logj"].shape != expected_shape or maps["disp"].shape != expected_shape:
            raise ValueError(f"{uid}: G0 Post-grid map shape differs from frozen expanded_bbox")


def build_frozen_contracts(cfg: dict[str, Any], split: str | None = None) -> list[FrozenSeriesContract]:
    """Build the authoritative 1011-case contract without accessing labels."""
    pairs = load_local_reference_pairs(cfg, split=split)
    raw_pairs = _pair_manifest_rows(cfg)
    g0_by_split = {name: _load_g0_table(cfg, name) for name in ("Train", "Valid")}
    g0_index = {
        name: {str(row["series_uid"]): row for row in table.to_dict("records")}
        for name, table in g0_by_split.items()
    }
    contracts: list[FrozenSeriesContract] = []
    for pair in sorted(pairs, key=lambda x: x.series_uid):
        raw = raw_pairs.get(pair.series_uid)
        row = g0_index[pair.split].get(pair.series_uid)
        if raw is None or row is None:
            raise KeyError(f"{pair.series_uid}: missing pair or G0 identity")
        case_dir = Path(cfg["paths"]["g0_registration_root"]) / pair.split.lower() / "cases" / pair.series_uid
        _validate_g0_case(pair, row, case_dir)
        pre_i = int(_number(row, "pre_peak_index", pair.series_uid))
        post_i = int(_number(row, "post_peak_index", pair.series_uid))
        pre_pos, pre_sources = _resolve_block(raw["pre_frozen_temporal_blocks_json"], pair.pre, pre_i)
        post_pos, post_sources = _resolve_block(raw["post_frozen_temporal_blocks_json"], pair.post, post_i)
        contracts.append(FrozenSeriesContract(
            series_uid=pair.series_uid, patient_id=_norm_id(pair.patient_id), split=pair.split,
            series_id=pair.series_id,
            pre=FrozenPhaseContract(pair.pre, pre_i, _number(row, "pre_peak_score", pair.series_uid), pre_pos, pre_sources),
            post=FrozenPhaseContract(pair.post, post_i, _number(row, "post_peak_score", pair.series_uid), post_pos, post_sources),
            g0_case_dir=case_dir, g0_row=row,
        ))
    expected = (int(cfg["expected"]["paired_series_train"]) + int(cfg["expected"]["paired_series_valid"])) if split is None else int(cfg["expected"]["paired_series_train" if split == "Train" else "paired_series_valid"])
    if len(contracts) != expected or len({c.series_uid for c in contracts}) != expected:
        raise AssertionError("frozen contract coverage/duplicate check failed")
    return contracts


def validate_case_inputs(contract: FrozenSeriesContract, *, stat_all_frames: bool = True) -> dict[str, Any]:
    """Validate frozen assets for one case; no frame discovery or reordering occurs."""
    payload: dict[str, Any] = {
        "series_uid": contract.series_uid, "patient_id": contract.patient_id, "split": contract.split,
        "g0_case_dir": str(contract.g0_case_dir), "valid": True, "reasons": [],
    }
    for phase in (contract.pre, contract.post):
        record = phase.record
        if not record.mask_path.is_file():
            payload["reasons"].append(f"{record.phase}_mask_missing")
        if stat_all_frames:
            missing = [path for path in record.frame_paths if not Path(path).is_file()]
            if missing:
                payload["reasons"].append(f"{record.phase}_missing_frames={len(missing)}")
        if len(phase.selected_block_positions) == 0 or phase.frozen_peak_index not in phase.selected_block_positions:
            payload["reasons"].append(f"{record.phase}_invalid_peak_block")
    payload["valid"] = not payload["reasons"]
    return payload


def select_smoke10(fov50_csv: str | Path) -> pd.DataFrame:
    table = pd.read_csv(fov50_csv, dtype=str, keep_default_na=False)
    required = {"series_uid", "patient_id", "stratum", "selection_rank"}
    if missing := sorted(required - set(table.columns)):
        raise KeyError(f"fov50 missing columns: {missing}")
    table["selection_rank_num"] = pd.to_numeric(table["selection_rank"], errors="raise")
    selected = (table.sort_values(["stratum", "selection_rank_num", "series_uid"])
                .groupby("stratum", sort=True, as_index=False, group_keys=False).head(2).copy())
    if selected["stratum"].nunique() != 5 or len(selected) != 10 or selected["series_uid"].duplicated().any():
        raise AssertionError("Smoke10 must be exactly 5 strata × 2 unique frozen series")
    return selected.drop(columns="selection_rank_num").sort_values(["stratum", "selection_rank", "series_uid"]).reset_index(drop=True)


def contract_rows(contracts: list[FrozenSeriesContract]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in contracts:
        row: dict[str, Any] = {
            "series_uid": item.series_uid, "patient_id": item.patient_id, "split": item.split,
            "series_id": item.series_id, "g0_case_dir": str(item.g0_case_dir),
        }
        for name, phase in (("pre", item.pre), ("post", item.post)):
            rec = phase.record
            row.update({
                f"{name}_phase_uid": rec.phase_uid, f"{name}_frame_paths": "|".join(rec.frame_paths),
                f"{name}_frame_list_hash": rec.frame_list_hash, f"{name}_n_frames": len(rec.frame_paths),
                f"{name}_expanded_bbox": rec.expanded_bbox.as_text(), f"{name}_mask_path": str(rec.mask_path),
                f"{name}_frozen_peak_index": phase.frozen_peak_index,
                f"{name}_frozen_peak_score": phase.frozen_peak_score,
                f"{name}_selected_block_positions": "|".join(map(str, phase.selected_block_positions)),
                f"{name}_selected_source_indices": "|".join(map(str, phase.frozen_source_indices)),
                f"{name}_resize_scale_x": rec.resize_scale_x, f"{name}_resize_scale_y": rec.resize_scale_y,
            })
        rows.append(row)
    return rows


def write_input_lock(cfg: dict[str, Any], run_root: str | Path, *, new_code_paths: list[Path]) -> dict[str, Any]:
    root = Path(run_root)
    g0 = Path(cfg["paths"]["g0_registration_root"])
    paths = {
        "pair_manifest": Path(cfg["paths"]["temporal_pairs"]),
        "roi_manifest": Path(cfg["paths"]["roi_phase_manifest"]),
        "g0_locked_config": g0 / "LOCKED_LOCAL_REFERENCE_REG_V1.yaml",
        "g0_train_table": g0 / "train_registration_qc.csv",
        "g0_valid_table": g0 / "valid_registration_qc.csv",
        "v5_registration_ants": Path(cfg["paths"]["v5_root"]) / "dsa_reg/registration_ants.py",
        "v5_registration_sitk": Path(cfg["paths"]["v5_root"]) / "dsa_reg/registration_sitk.py",
        "v5_preprocessing": Path(cfg["paths"]["v5_root"]) / "dsa_reg/preprocessing.py",
        "v5_hemodynamics": Path(cfg["paths"]["v5_root"]) / "dsa_reg/hemodynamics.py",
        "v5_features": Path(cfg["paths"]["v5_root"]) / "dsa_reg/features.py",
        "fov50_series": Path(cfg["paths"]["fov50_series"]),
    }
    payload = {
        "contract_version": "local_reference_jacobian_hemo_20260820",
        "outcome_accessed": False,
        "g0_rigid_or_syn_rerun": False,
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "new_code_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path) for path in new_code_paths},
    }
    atomic_json(payload, root / "contracts" / "INPUT_LOCK.json")
    return payload
