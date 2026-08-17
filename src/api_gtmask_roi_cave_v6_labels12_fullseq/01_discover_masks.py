#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return path.name.casefold().endswith(NIFTI_SUFFIXES)


def normalized_tokens(path: Path) -> str:
    return "/".join(part.casefold() for part in path.parts)


def is_mask_file(path: Path) -> bool:
    name = path.name.casefold()
    stem = name[:-7] if name.endswith(".nii.gz") else name[:-4] if name.endswith(".nii") else name
    if any(token in stem for token in ("segmentation", "segment", "mask", "label")):
        return True
    return bool(re.search(r"(^|[-_])(seg|roi)([-_]|$)", stem))


def is_reference_file(path: Path) -> bool:
    if not is_nifti(path) or is_mask_file(path):
        return False
    name = path.name.casefold()
    return any(token in name for token in ("image", "reference", "ref", "img"))


def phase_hint(path: Path) -> str:
    text = normalized_tokens(path)
    pre = bool(re.search(r"(^|[/_\-])pre([/_\-]|$)", text))
    post = bool(re.search(r"(^|[/_\-])post([/_\-]|$)", text))
    if pre and not post:
        return "pre"
    if post and not pre:
        return "post"
    return ""


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, FileNotFoundError, OSError):
        return False


def deepest_series_hint(path: Path, patient_phases: pd.DataFrame) -> str:
    candidates: list[tuple[int, str, Path, Path]] = []
    for record in patient_phases.drop_duplicates("series_uid").to_dict("records"):
        series_text = str(record.get("series_path", "")).strip()
        patient_text = str(record.get("source_medical_record_root", "")).strip()
        if not series_text:
            continue
        series_root = Path(series_text).expanduser()
        patient_root = Path(patient_text).expanduser() if patient_text else series_root
        if path_is_within(path, series_root):
            candidates.append((len(series_root.parts), str(record["series_uid"]), series_root, patient_root))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, uid, series_root, patient_root = candidates[0]
    # 多 series 患者中，患者根目录不能自动等价于 main series。
    if patient_phases["series_uid"].nunique() > 1:
        try:
            if series_root.resolve() == patient_root.resolve():
                return ""
        except (FileNotFoundError, OSError):
            pass
    return uid


def api_phase_hint(path: Path, patient_phases: pd.DataFrame) -> str:
    hits: set[str] = set()
    for record in patient_phases.to_dict("records"):
        phase = str(record["phase"])
        api_dir = str(record.get("api_dir", "")).strip()
        if api_dir and path_is_within(path, Path(api_dir).expanduser()):
            hits.add(phase)
    return next(iter(hits)) if len(hits) == 1 else ""




def minimize_roots(roots: set[Path]) -> list[Path]:
    """Keep only highest-level existing roots to avoid recursively scanning nested trees repeatedly."""
    existing = sorted({root for root in roots if root.is_dir()}, key=lambda item: (len(item.parts), str(item)))
    kept: list[Path] = []
    for root in existing:
        if any(path_is_within(root, parent) for parent in kept):
            continue
        kept.append(root)
    return kept

def find_reference(mask: Path, declared_phase: str) -> Path | None:
    candidates = [p for p in mask.parent.iterdir() if p.is_file() and is_reference_file(p)]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, int, str]:
        hint = phase_hint(path)
        same_phase = int(bool(declared_phase and hint == declared_phase))
        generic_image = int(path.name.casefold() in {"image.nii", "image.nii.gz"})
        return (-same_phase, -generic_image, path.name.casefold())

    return sorted(candidates, key=rank)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    phase_index = pd.read_csv(manifests / "source_phase_index_all.csv", dtype=str, keep_default_na=False)
    by_patient = {pid: group.copy() for pid, group in phase_index.groupby("patient_id", sort=False)}

    extra_roots = [Path(value).expanduser() for value in cfg.get("annotation", {}).get("extra_search_roots", [])]
    inventory: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for patient_id, patient_phases in by_patient.items():
        roots: set[Path] = set()
        for record in patient_phases.to_dict("records"):
            for field in ("source_medical_record_root", "series_path", "api_dir"):
                value = str(record.get(field, "")).strip()
                if value:
                    roots.add(Path(value).expanduser())
        for base in extra_roots:
            candidate = base / patient_id
            if candidate.is_dir():
                roots.add(candidate)

        found: set[Path] = set()
        scan_roots = minimize_roots(roots)
        for root in scan_roots:
            try:
                for path in root.rglob("*"):
                    if path.is_file() and is_nifti(path) and is_mask_file(path):
                        found.add(path)
            except (PermissionError, OSError) as exc:
                errors.append({"patient_id": patient_id, "root": str(root), "reason": f"{type(exc).__name__}:{exc}"})

        for mask in sorted(found):
            try:
                resolved = str(mask.resolve())
            except OSError:
                resolved = str(mask.absolute())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            explicit = phase_hint(mask)
            api_hint = api_phase_hint(mask, patient_phases)
            declared = explicit or api_hint
            series_hint = deepest_series_hint(mask, patient_phases)
            reference = find_reference(mask, declared)
            mask_sha = sha256_file(mask)
            inventory.append({
                "mask_uid": mask_sha,
                "patient_id": patient_id,
                "split": str(patient_phases.iloc[0]["split"]),
                "mask_path": str(mask),
                "mask_sha256": mask_sha,
                "mask_filename": mask.name,
                "mask_parent": str(mask.parent),
                "phase_hint": declared,
                "phase_hint_source": "path_token" if explicit else "api_directory" if api_hint else "",
                "series_hint_uid": series_hint,
                "reference_image_path": str(reference) if reference else "",
                "reference_sha256": sha256_file(reference) if reference else "",
                "storage_layout": "image_plus_segmentation" if reference else "standalone_segmentation",
            })

    frame = pd.DataFrame(inventory)
    if frame.empty:
        raise RuntimeError("没有发现任何 NIfTI Mask 文件")
    frame = frame.sort_values(["split", "patient_id", "mask_path"]).reset_index(drop=True)
    # 相同内容的副本仍保留路径信息，但标记重复组，映射时只选一个主路径。
    frame["same_content_count"] = frame.groupby("mask_sha256")["mask_path"].transform("count")
    atomic_csv(frame, manifests / "mask_inventory.csv")
    atomic_csv(pd.DataFrame(errors), reports / "01_mask_discovery_errors.csv")
    summary = {
        "mask_paths": int(len(frame)),
        "unique_mask_contents": int(frame["mask_sha256"].nunique()),
        "patients_with_masks": int(frame["patient_id"].nunique()),
        "image_plus_segmentation": int((frame["storage_layout"] == "image_plus_segmentation").sum()),
        "standalone_segmentation": int((frame["storage_layout"] == "standalone_segmentation").sum()),
        "phase_pre": int((frame["phase_hint"] == "pre").sum()),
        "phase_post": int((frame["phase_hint"] == "post").sum()),
        "phase_unknown": int((frame["phase_hint"] == "").sum()),
        "exact_series_hint": int(frame["series_hint_uid"].astype(bool).sum()),
        "scan_errors": len(errors),
    }
    atomic_json(summary, reports / "01_mask_discovery_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
