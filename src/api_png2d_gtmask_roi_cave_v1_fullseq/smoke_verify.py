#!/usr/bin/env python3
"""Smoke verification for eligible Local-CAVE extraction.

Per extracted smoke phase this prints and writes the full audit trail
(phase_uid, frame_list_hash, mask, orientation, bboxes, shapes, temporal
indices) and asserts the extraction invariants:

1. local_shape matches the phase's own used bbox;
2. the model really received local frames (local embedding != whole
   embedding for the same phase);
3. one ROI per phase (single used_bbox, matching the ROI manifest);
4. different phases use their own ROI/mask (no cross-phase reuse);
5. Whole and Local temporal view indices are identical;
6. checkpoint, feature schema and temporal policy are unchanged;
7. outputs live in the independent local featurebank;
8. no full-sequence local JPGs were saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import as_bool, atomic_json, load_config, sha256_file


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def view_indices_of(metadata: dict) -> list[dict]:
    return [
        {"indices": block.get("indices"), "view_indices": block.get("view_indices")}
        for block in metadata.get("blocks", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    smoke_root = Path(cfg["paths"]["outputs"]) / "smoke_local_eligible_featurebank"
    whole_root = Path(cfg["whole_featurebank"])
    frozen = json.loads((Path(cfg["paths"]["outputs"]) / "cave_frozen_configs" / f"local_full_{args.split.casefold()}.json").read_text(encoding="utf-8"))
    model_input_shape = int(frozen.get("image_size", 0))
    if frozen.get("foreground_rule") != "labels_in_1_2_equivalent_nonzero":
        raise AssertionError("Frozen CAVE config does not declare labels_in_1_2_equivalent_nonzero")
    if frozen.get("selected_labels") != [1, 2]:
        raise AssertionError("Frozen CAVE config does not declare selected_labels=[1,2]")
    output_text = str(smoke_root)
    if "api_png2d_gtmask_roi_cave_v1_fullseq" not in output_text:
        raise AssertionError("PNG2D smoke output is outside the isolated output root")
    if (
        "api_gtmask_roi_cave_v5_fullmask_fullseq" in output_text
        or "api_gtmask_roi_cave_v6_labels12_fullseq" in output_text
    ):
        raise AssertionError("PNG2D smoke output points to an old NIfTI pipeline root")

    roi = pd.read_csv(manifests / "roi_phase_manifest_eligible.csv", dtype=str, keep_default_na=False)
    roi = roi[roi["split"] == args.split]
    roi_by_uid = {str(r.phase_uid): r for r in roi.itertuples(index=False)}

    def scientific_schema(path: Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: value for key, value in payload.items() if key not in {"frozen_config_hash", "schema_sha256"}}

    local_schema_path = smoke_root / "feature_schema.json"
    whole_schema_path = whole_root / "feature_schema.json"
    # smoke root 的 schema 文件由先跑的那个 split 写入；本 split 应以自己 shard 的 schema 为准
    shard_candidates = sorted((Path(cfg["paths"]["outputs"]) / f"shards_{args.split.casefold()}_smoke").glob("shard_*/feature_schema.json"))
    split_schema_path = shard_candidates[0] if shard_candidates else local_schema_path
    # CAVE 的 feature_schema_sha256 是 schema 文件内部的自校验字段，不是文件字节 sha
    local_schema_self_sha = json.loads(split_schema_path.read_text(encoding="utf-8")).get("schema_sha256", "")

    base = smoke_root / args.split.casefold()
    failures: list[str] = []
    rows: list[dict] = []
    success_files = sorted(base.rglob(".SUCCESS.json")) if base.is_dir() else []
    if not success_files:
        raise RuntimeError(f"smoke featurebank 为空：{base}")

    for success_path in success_files:
        phase_dir = success_path.parent
        success = json.loads(success_path.read_text(encoding="utf-8"))
        metadata = json.loads((phase_dir / "metadata.json").read_text(encoding="utf-8"))
        roi_meta = metadata.get("roi", {})
        phase_uid = str(roi_meta.get("phase_uid", success.get("phase_uid", "")))
        record = roi_by_uid.get(phase_uid)
        if record is None:
            failures.append(f"{phase_uid}: 不在 eligible ROI manifest 中")
            continue

        used_bbox = str(roi_meta.get("used_bbox", ""))
        side = int(used_bbox.split("|")[2]) - int(used_bbox.split("|")[0]) if used_bbox.count("|") == 3 else -1
        whole_dir = whole_root / args.split.casefold() / str(record.patient_id) / str(record.series_uid) / str(record.phase)
        whole_metadata = json.loads((whole_dir / "metadata.json").read_text(encoding="utf-8"))
        whole_success = json.loads((whole_dir / ".SUCCESS.json").read_text(encoding="utf-8"))

        local_embedding = np.load(phase_dir / "embedding_5120.npy")
        whole_embedding = np.load(whole_dir / "embedding_5120.npy")
        similarity = cosine(local_embedding, whole_embedding)

        row = {
            "phase_uid": phase_uid,
            "frame_list_hash": str(record.frame_list_hash),
            "annotation_source": str(roi_meta.get("annotation_source", "")),
            "mask_path": str(record.mask_path),
            "mask_resized_to_frame": bool(roi_meta.get("mask_resized_to_frame", False)),
            "effective_mask_array_sha256": str(roi_meta.get("effective_mask_array_sha256", "")),
            "orientation_transform": str(roi_meta.get("orientation_transform", "")),
            "foreground_rule": str(roi_meta.get("foreground_rule", "")),
            "selected_labels": json.dumps(roi_meta.get("selected_labels", []), separators=(",", ":")),
            "labels_present": json.dumps(roi_meta.get("labels_present", []), separators=(",", ":")),
            "labels_ignored": json.dumps(roi_meta.get("labels_ignored", []), separators=(",", ":")),
            "selected_foreground_pixels": int(roi_meta.get("selected_foreground_pixels", 0)),
            "ignored_foreground_pixels": int(roi_meta.get("ignored_foreground_pixels", 0)),
            "original_bbox": str(roi_meta.get("original_bbox", "")),
            "expanded_bbox": str(roi_meta.get("primary_bbox", "")),
            "used_bbox": used_bbox,
            "fallback_used": bool(roi_meta.get("fallback_used", False)),
            "fallback_level": str(
                roi_meta.get(
                    "fallback_level",
                    "fallback" if bool(roi_meta.get("fallback_used", False)) else "primary",
                )
            ),
            "original_shape": "x".join(map(str, metadata.get("original_shape", []))),
            "local_shape": f"{side}x{side}",
            "model_input_shape": f"{model_input_shape}x{model_input_shape}",
            "whole_temporal_indices": json.dumps(view_indices_of(whole_metadata), sort_keys=True),
            "local_temporal_indices": json.dumps(view_indices_of(metadata), sort_keys=True),
            "embedding_cosine_local_vs_whole": similarity,
            "input_mosaic": str(phase_dir / "input_mosaic.jpg"),
        }
        rows.append(row)

        # 1. local_shape 来自本 phase 的 bbox
        expected_by_level = {
            "primary": str(record.expanded_bbox),
            "fallback": str(record.fallback_bbox),
            "extended": str(record.extended_fallback_bbox),
        }
        expected = expected_by_level.get(row["fallback_level"], "")
        if roi_meta.get("annotation_source") != "png_2d_gt":
            failures.append(f"{phase_uid}: annotation_source 不是 png_2d_gt")
        if roi_meta.get("effective_mask_array_sha256") != str(record.effective_mask_array_sha256):
            failures.append(f"{phase_uid}: effective mask SHA 与 ROI manifest 不一致")
        if bool(roi_meta.get("mask_resized_to_frame", False)) != as_bool(record.mask_resized_to_frame):
            failures.append(f"{phase_uid}: mask resize provenance 与 ROI manifest 不一致")
        if roi_meta.get("target_rule") != "png2d_gt_labels_1_2_nonzero":
            failures.append(f"{phase_uid}: target_rule 不是 png2d_gt_labels_1_2_nonzero")
        if roi_meta.get("foreground_rule") != "labels_in_1_2_equivalent_nonzero":
            failures.append(f"{phase_uid}: foreground_rule 不是 labels_in_1_2_equivalent_nonzero")
        if roi_meta.get("selected_labels") != [1, 2]:
            failures.append(f"{phase_uid}: selected_labels 不是 [1,2]")
        if int(roi_meta.get("selected_foreground_pixels", 0)) <= 0:
            failures.append(f"{phase_uid}: selected_foreground_pixels 非正")
        if local_embedding.shape != (5120,) or not np.isfinite(local_embedding).all():
            failures.append(f"{phase_uid}: Local embedding shape/finite 失败: {local_embedding.shape}")
        if not (phase_dir / "input_mosaic.jpg").is_file():
            failures.append(f"{phase_uid}: 缺少 input_mosaic.jpg")
        if used_bbox != expected:
            failures.append(f"{phase_uid}: used_bbox 与 ROI manifest 不一致")
        if side <= 0:
            failures.append(f"{phase_uid}: 非法 local_shape")
        # 2. 模型确实收到局部帧（embedding 与 whole 显著不同）
        if similarity >= 0.9995:
            failures.append(f"{phase_uid}: Local/Whole embedding 几乎相同（cosine={similarity:.6f}），模型可能仍收到整图")
        if success.get("embedding_sha256") == whole_success.get("embedding_sha256"):
            failures.append(f"{phase_uid}: Local/Whole embedding_sha256 相同")
        # 3. 同 phase 单一 ROI
        if used_bbox.count("|") != 3:
            failures.append(f"{phase_uid}: used_bbox 非法")
        # 5. 时间索引一致
        if row["whole_temporal_indices"] != row["local_temporal_indices"]:
            failures.append(f"{phase_uid}: Whole/Local 时间索引不一致")
        # 6. checkpoint/schema/temporal policy 不变
        if success.get("checkpoint_sha256") != whole_success.get("checkpoint_sha256"):
            failures.append(f"{phase_uid}: checkpoint 与 Whole 不一致")
        # schema 科学内容必须与 Whole 一致；frozen_config_hash 因 ROI provenance 合法不同
        if success.get("feature_schema_sha256") != local_schema_self_sha:
            failures.append(f"{phase_uid}: feature schema 与 Local featurebank 不一致")
        if roi_meta.get("temporal_policy") != cfg.get("temporal", {}).get("policy", "freeze_whole_indices"):
            failures.append(f"{phase_uid}: temporal policy 不一致")
        # 7. 输出在独立 featurebank
        if str(smoke_root) == str(whole_root) or str(phase_dir).startswith(str(whole_root)):
            failures.append(f"{phase_uid}: 输出写入了 Whole featurebank")
        # 8. 不保存全量局部 JPG
        if success.get("local_frames_saved") or roi_meta.get("local_frames_saved"):
            failures.append(f"{phase_uid}: local_frames_saved=True")

    # 6b. Local 与 Whole 的科学 feature schema 一致（网络/归一化/pooling 未改）
    if scientific_schema(split_schema_path) != scientific_schema(whole_schema_path):
        failures.append("Local/Whole 科学 feature schema 不一致")

    # 4. 不同 phase 使用各自 ROI/Mask
    mask_by_phase = {row["phase_uid"]: row["mask_path"] for row in rows}
    if len(set(mask_by_phase.values())) != len(mask_by_phase):
        failures.append("存在不同 phase 复用同一 Mask 的情况")

    summary = {
        "split": args.split,
        "smoke_root": str(smoke_root),
        "phases_checked": len(rows),
        "failures": failures,
        "status": "failed" if failures else "success",
        "rows": rows,
    }
    atomic_json(summary, reports / f"smoke_verify_{args.split.casefold()}.json")
    printable = [{k: (v if k not in {"whole_temporal_indices", "local_temporal_indices"} else "...") for k, v in row.items()} for row in rows]
    print(json.dumps({**summary, "rows": printable}, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
