#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from assets import (
    TRANSFORMS, affine_equal, apply_orientation, build_summary_channels, draw_overlay, find_reference_match,
    lesion_and_context_masks, load_nifti_image, load_nifti_mask, mask_contrast_score, read_frames,
)
from common import atomic_csv, atomic_json, configure_runtime, load_config, parse_pipe, sha256_file, stage_logger, write_marker


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--max-rows", type=int); args = parser.parse_args()
    config = load_config(args.config); configure_runtime(config)
    finish = stage_logger("03_infer_reference_rule_and_alignment")
    manifests = Path(config["paths"]["manifests"]); reports = Path(config["paths"]["reports"])
    source_path = manifests / "authoritative_roi_manifest_candidates.csv"
    frame = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    frame = frame[frame["primary_candidate"].astype(str).str.casefold().eq("true")].copy()
    if args.max_rows: frame = frame.head(args.max_rows).copy()
    alignment_cfg = config["alignment"]; mask_cfg = config["mask"]
    results: dict[str, dict] = {}
    transform_votes: Counter[str] = Counter()
    a_scores = []
    for row in frame[frame["annotation_grade_pre_alignment"] == "A"].to_dict("records"):
        try:
            paths = parse_pipe(row["frame_paths"]); frames = read_frames(paths)
            reference, reference_info = load_nifti_image(Path(row["reference_image_path"]))
            mask, mask_info = load_nifti_mask(Path(row["segmentation_path"]))
            match = find_reference_match(reference, frames, alignment_cfg.get("transforms", TRANSFORMS))
            reference_mask_shape_match=reference.shape==mask.shape
            reference_mask_affine_match=affine_equal(reference_info,mask_info)
            oriented_mask=apply_orientation(mask,match["orientation_transform"])
            mask_frame_shape_match=oriented_mask.shape==frames.shape[1:]
            if mask_frame_shape_match:
                lesion,context,_=lesion_and_context_masks(oriented_mask,mask_cfg["lesion_labels"],mask_cfg["context_labels"])
                overlay=mask_contrast_score(build_summary_channels(frames)["max_enhancement"],lesion,context)
            else: overlay={"score":float("-inf"),"lesion_z":float("nan"),"context_z":float("nan")}
            base_qc = bool(match["score"] >= float(alignment_cfg["reference_accept_score"])
                and reference_mask_shape_match and reference_mask_affine_match and mask_frame_shape_match
                and overlay["score"] >= float(alignment_cfg["minimum_mask_contrast_z"]))
            strict_margin=match["orientation_margin"] >= float(alignment_cfg["reference_accept_margin"])
            exact_reference_fallback=bool(match["score"]>=0.995 and match["ncc"]>=0.995 and match["mae"]<=0.01)
            accepted=base_qc and (strict_margin or exact_reference_fallback)
            result = {
                "phase_uid": row["phase_uid"], "alignment_status": ("accepted_reference" if strict_margin else "accepted_reference_exact_fallback") if accepted else "failed_reference",
                "orientation_transform": match["orientation_transform"], "alignment_score": match["score"],
                "alignment_margin": match["margin"], "matched_reference_frame_position": match["frame_position"],
                "matched_reference_frame_path": paths[int(match["frame_position"])],
                "reference_ncc": match["ncc"], "reference_ssim": match["ssim"], "reference_mae": match["mae"],
                "reference_frame_margin":match["frame_match_margin"],"reference_orientation_margin":match["orientation_margin"],
                "exact_reference_fallback":exact_reference_fallback and not strict_margin,
                "reference_mask_shape_match":reference_mask_shape_match,"reference_mask_affine_match":reference_mask_affine_match,"mask_frame_shape_match":mask_frame_shape_match,
                "lesion_contrast_z":overlay["lesion_z"],"context_contrast_z":overlay["context_z"],"overlay_contrast_score":overlay["score"],
                "alignment_error": "", "reference_info_json": json.dumps(reference_info, sort_keys=True, default=float),
                "mask_info_json":json.dumps(mask_info,sort_keys=True,default=float),
            }
            results[row["phase_uid"]] = result
            if accepted:
                transform_votes[match["orientation_transform"]] += 1; a_scores.append(float(match["score"]))
        except Exception as exc:
            results[row["phase_uid"]] = {"phase_uid": row["phase_uid"], "alignment_status": "failed_reference_exception", "orientation_transform": "", "alignment_score": np.nan, "alignment_margin": np.nan, "alignment_error": repr(exc)}
    if not transform_votes:
        raise RuntimeError("No accepted A-grade reference alignment; refusing to guess B-grade orientation")
    global_transform, global_votes = transform_votes.most_common(1)[0]
    global_fraction = global_votes / max(sum(transform_votes.values()), 1)
    for row in frame[frame["annotation_grade_pre_alignment"] != "A"].to_dict("records"):
        try:
            paths = parse_pipe(row["frame_paths"]); frames = read_frames(paths)
            mask, mask_info = load_nifti_mask(Path(row["segmentation_path"]))
            summaries = build_summary_channels(frames); enhancement = summaries["max_enhancement"]
            global_oriented=apply_orientation(mask,global_transform); b_mask_resized_to_frames=False
            if global_oriented.shape!=enhancement.shape:
                source_ratio=global_oriented.shape[1]/max(global_oriented.shape[0],1); target_ratio=enhancement.shape[1]/max(enhancement.shape[0],1)
                if abs(source_ratio-target_ratio)>0.02: raise AssertionError(f"B mask/frame aspect mismatch {global_oriented.shape} {enhancement.shape}")
                global_oriented=cv2.resize(global_oriented,(enhancement.shape[1],enhancement.shape[0]),interpolation=cv2.INTER_NEAREST); b_mask_resized_to_frames=True
            global_lesion,global_context,_=lesion_and_context_masks(global_oriented,mask_cfg["lesion_labels"],mask_cfg["context_labels"])
            global_item={"transform":global_transform,**mask_contrast_score(enhancement,global_lesion,global_context)}
            side=192; enhancement_small=cv2.resize(enhancement.astype(np.float32),(side,side),interpolation=cv2.INTER_AREA); candidates=[]
            for transform in alignment_cfg.get("transforms", TRANSFORMS):
                oriented=apply_orientation(mask,transform)
                if oriented.shape!=enhancement.shape:
                    source_ratio=oriented.shape[1]/max(oriented.shape[0],1); target_ratio=enhancement.shape[1]/max(enhancement.shape[0],1)
                    if abs(source_ratio-target_ratio)>0.02: continue
                    oriented=cv2.resize(oriented,(enhancement.shape[1],enhancement.shape[0]),interpolation=cv2.INTER_NEAREST)
                oriented_small=cv2.resize(oriented,(side,side),interpolation=cv2.INTER_NEAREST)
                lesion,context,_=lesion_and_context_masks(oriented_small,mask_cfg["lesion_labels"],mask_cfg["context_labels"])
                candidates.append({"transform":transform,**mask_contrast_score(enhancement_small,lesion,context,kernel_size=7)})
            candidates.sort(key=lambda item:item["score"],reverse=True)
            global_thumbnail=next((item for item in candidates if item["transform"]==global_transform),None); best=candidates[0] if candidates else None
            score_gap=float(best["score"]-global_thumbnail["score"]) if best and global_thumbnail else float("inf")
            accepted=bool(global_item["score"]>=float(alignment_cfg["minimum_mask_contrast_z"]) and score_gap<=float(alignment_cfg["maximum_global_transform_score_gap"]))
            results[row["phase_uid"]] = {
                "phase_uid": row["phase_uid"], "alignment_status": "accepted_global_transform" if accepted else "failed_b_alignment",
                "orientation_transform": global_transform if accepted else "",
                "alignment_score": global_item["score"] if global_item else np.nan,
                "alignment_margin": -score_gap if np.isfinite(score_gap) else np.nan,
                "best_b_transform": best["transform"] if best else "", "best_b_score": best["score"] if best else np.nan,
                "global_thumbnail_score":global_thumbnail["score"] if global_thumbnail else np.nan,"direction_score_gap":score_gap,"b_mask_resized_to_frames":b_mask_resized_to_frames,
                "lesion_contrast_z": global_item["lesion_z"] if global_item else np.nan,
                "context_contrast_z": global_item["context_z"] if global_item else np.nan,
                "alignment_error": "", "mask_info_json": json.dumps(mask_info, sort_keys=True, default=float),
            }
        except Exception as exc:
            results[row["phase_uid"]] = {"phase_uid": row["phase_uid"], "alignment_status": "failed_b_exception", "orientation_transform": "", "alignment_score": np.nan, "alignment_margin": np.nan, "alignment_error": repr(exc)}
    aligned = frame.merge(pd.DataFrame(results.values()), on="phase_uid", how="left", validate="one_to_one")
    aligned["alignment_accepted"] = aligned["alignment_status"].astype(str).str.startswith("accepted")
    aligned["annotation_grade"] = aligned["annotation_grade_pre_alignment"]
    primary = aligned[aligned["alignment_accepted"]].copy()
    excluded = aligned[~aligned["alignment_accepted"]].copy()
    excluded["exclusion_reason"] = "alignment:" + excluded["alignment_status"].astype(str)
    atomic_csv(aligned, manifests / "authoritative_roi_manifest_all.csv")
    atomic_csv(primary, manifests / "authoritative_roi_manifest_primary.csv")
    atomic_csv(excluded, manifests / "excluded_assets.csv")
    qc_root = reports / "alignment_qc"
    regular=primary.sort_values("phase_uid").head(int(alignment_cfg.get("qc_examples",50)))
    low_conf=aligned[(~aligned["alignment_accepted"])|aligned["alignment_status"].astype(str).str.contains("fallback",case=False,na=False)]
    examples=pd.concat([regular,low_conf],ignore_index=True).drop_duplicates("phase_uid")
    qc_written=0
    for row in examples.to_dict("records"):
        try:
            frames = read_frames(parse_pipe(row["frame_paths"])); summary = build_summary_channels(frames)["max_enhancement"]
            transform=row.get("orientation_transform","") or row.get("best_b_transform","")
            if not transform: continue
            mask, _ = load_nifti_mask(Path(row["segmentation_path"])); mask = apply_orientation(mask, transform)
            if mask.shape == summary.shape:
                draw_overlay(summary, mask, qc_root / f"{row['phase_uid']}.jpg", f"{row['patient_id']} {row['series_id']} {row['phase']} {row['annotation_grade']} {row['alignment_status']}"); qc_written+=1
        except Exception:
            pass
    summary = {
        "candidate_rows": len(frame), "accepted_rows": len(primary), "excluded_rows": len(excluded),
        "accepted_patients": int(primary["patient_id"].nunique()),
        "accepted_by_split_phase": primary.groupby(["split", "phase"]).size().to_dict(),
        "accepted_by_grade": primary["annotation_grade"].value_counts().to_dict(),
        "alignment_status": aligned["alignment_status"].value_counts().to_dict(),
        "global_orientation_transform": global_transform, "global_transform_vote_fraction": global_fraction,
        "a_reference_score_median": float(np.median(a_scores)) if a_scores else None,
        "source_manifest_sha256": sha256_file(source_path),
        "label_protocol_status": mask_cfg["label_protocol_status"],
        "qc_requested":len(examples),"qc_written":qc_written,
    }
    summary["accepted_by_split_phase"] = {"|".join(key): int(value) for key, value in summary["accepted_by_split_phase"].items()}
    atomic_json(summary, reports / "authoritative_manifest_summary.json")
    atomic_csv(aligned[[c for c in aligned.columns if c.startswith("alignment") or c in {"phase_uid","patient_id","split","series_uid","phase","annotation_grade","orientation_transform","reference_ncc","reference_ssim","reference_mae","best_b_transform","best_b_score","lesion_contrast_z","context_contrast_z"}]], reports / "alignment_audit.csv")
    write_marker(reports / ".ASSET_SUCCESS", "03_infer_reference_rule_and_alignment", config, {"source_manifest_sha256": sha256_file(source_path)}, summary)
    finish(summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
