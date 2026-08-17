#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_json, atomic_text, configure_runtime, load_config, stage_logger, write_marker


def cave_audit(feature_root:Path,manifest_path:Path):
    manifest=pd.read_csv(manifest_path,dtype=str,keep_default_na=False); expected=[]
    for row in manifest.to_dict("records"):
        for phase in ("pre","post"):
            if str(row.get(f"can_run_{phase}","")).casefold()=="true":
                expected.append((str(row["split"]),str(row["patient_id"]),str(row["series_uid"]),phase))
    norms=[]; vessel=[]; invalid=0; successes=0; failed=[]
    for split,patient_id,series_uid,phase in expected:
        directory=feature_root/split.casefold()/patient_id/series_uid/phase
        try:
            if not (directory/".SUCCESS.json").is_file():
                failed.append({"patient_id":patient_id,"series_uid":series_uid,"phase":phase,"reason":"missing_success"}); continue
            embedding=np.load(directory/"embedding_5120.npy"); qc=json.loads((directory/"qc.json").read_text(encoding="utf-8"));
            if embedding.shape!=(5120,) or not np.isfinite(embedding).all():
                invalid+=1; failed.append({"patient_id":patient_id,"series_uid":series_uid,"phase":phase,"reason":"invalid_embedding"}); continue
            successes+=1; norms.append(float(np.linalg.norm(embedding))); vessel.append(float(qc.get("vessel_probability_mean_fov",np.nan)))
        except Exception as exc:
            invalid+=1; failed.append({"patient_id":patient_id,"series_uid":series_uid,"phase":phase,"reason":repr(exc)})
    planned=len(expected)
    return {"planned_phases":planned,"success_phases":successes,"failure_rate":(planned-successes)/max(planned,1),"invalid_embedding_phases":invalid,"failed_phases":failed,"embedding_norm_median":float(np.nanmedian(norms)) if norms else None,"vessel_probability_mean_fov_median":float(np.nanmedian(vessel)) if vessel else None}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); args=parser.parse_args(); config=load_config(args.config); configure_runtime(config); finish=stage_logger("14_summarize_pipeline")
    reports=Path(config["paths"]["reports"]); outputs=Path(config["paths"]["outputs"]); manifests=Path(config["paths"]["manifests"])
    asset=json.loads((reports/"authoritative_manifest_summary.json").read_text(encoding="utf-8")); seg=json.loads((reports/"segmentation_oof_training_summary.json").read_text(encoding="utf-8")); task=json.loads((reports/"adverse_task_audit.json").read_text(encoding="utf-8")); roi=json.loads((reports/"roi_manifest_summary.json").read_text(encoding="utf-8"))
    ablation=json.loads((reports/"ablation_summary.json").read_text(encoding="utf-8")) if (reports/"ablation_summary.json").is_file() else {"status":"not_run_due_to_failed_gate"}
    gate_cfg=config["gates"]; phase_gate={}
    for phase,payload in seg["phases"].items():
        metrics=payload["threshold_metrics"]; phase_gate[phase]={"on_target":float(metrics["on_target"]),"coverage":float(metrics["coverage"]),"empty":float(metrics["empty"]),"passed":float(metrics["on_target"])>=gate_cfg["minimum_oof_on_target"] and float(metrics["coverage"])>=gate_cfg["minimum_padded_box_coverage"] and float(metrics["empty"])<=gate_cfg["maximum_empty_rate"]}
    segmentation_gate=bool(phase_gate) and all(item["passed"] for item in phase_gate.values())
    cave={}
    for branch in ("pred","gt","all_nonzero"):
        cave[branch]={}
        for split in ("Train","Valid"):
            cave[branch][split]=cave_audit(outputs/f"cave_{branch}_roi_featurebank",manifests/f"cave_manifest_{branch}_{split.casefold()}.csv")
    cave_gate=all(item["failure_rate"]<=gate_cfg["maximum_cave_phase_failure_rate"] and item["invalid_embedding_phases"]==0 for branch in cave.values() for item in branch.values())
    technical_complete=(reports/".ABLATIONS_SUCCESS").is_file(); primary_success=technical_complete and task["eligible_for_models"] and segmentation_gate and cave_gate
    selected=pd.read_csv(reports/"selected_model_metrics.csv").to_dict("records") if (reports/"selected_model_metrics.csv").is_file() else []
    bootstrap=pd.read_csv(reports/"whole_vs_roi_paired_bootstrap.csv").to_dict("records") if (reports/"whole_vs_roi_paired_bootstrap.csv").is_file() else []
    selected_ci=pd.read_csv(reports/"all_ablation_selected_bootstrap_ci.csv").query("ablation == 'pred_roi'").to_dict("records") if (reports/"all_ablation_selected_bootstrap_ci.csv").is_file() else []
    summary={"version":config["version"],"technical_pipeline_complete":technical_complete,"pred_roi_primary_success":primary_success,"asset_summary":asset,"segmentation_gate":{"passed":segmentation_gate,"phases":phase_gate},"cave_gate":{"passed":cave_gate,"audit":cave},"adverse_task":task,"roi_summary":roi,"selected_pred_roi_metrics":selected,"selected_pred_roi_bootstrap_ci":selected_ci,"whole_vs_roi_bootstrap":bootstrap,"ablation_summary":ablation,"label_protocol_status":config["mask"]["label_protocol_status"],"whole_baseline_definition":"same patient cohort, same folds, corrected trainer","gt_roi_definition":"oracle only","pred_roi_definition":"Train OOF masks and Valid all-Train frozen segmentation models"}
    atomic_json(cave["pred"]["Train"],reports/"cave_train_audit.json"); atomic_json(cave["pred"]["Valid"],reports/"cave_valid_audit.json"); atomic_json(summary,reports/"final_summary.json")
    lines=["# API adverse lesion CAVE v1 final summary","",f"- Technical pipeline complete: `{technical_complete}`",f"- Predicted ROI primary gate passed: `{primary_success}`",f"- Segmentation gate: `{segmentation_gate}`",f"- CAVE gate: `{cave_gate}`",f"- Train adverse patients: `{task['train_rows']}`; positives: `{task['train_positive']}`",f"- Valid adverse patients: `{task['valid_rows']}`; positives: `{task['valid_positive']}`",f"- ROI fallbacks: `{roi['fallbacks']}`",f"- Label protocol: `{config['mask']['label_protocol_status']}`","","## Branch definitions","","- Whole baseline: frozen whole-image CAVE, retrained on the exact ROI cohort and folds.","- GT ROI: oracle upper bound using ground-truth masks.","- Predicted ROI: deployable branch using Train OOF masks and Valid frozen-model masks.","","## Selected predicted-ROI metrics","",pd.DataFrame(selected).to_markdown(index=False) if selected else "No selected metrics.","","## Whole vs ROI paired bootstrap","",pd.DataFrame(bootstrap).to_markdown(index=False) if bootstrap else "No bootstrap results."]
    lines.extend(["","## Selected predicted-ROI bootstrap 95% CI","",pd.DataFrame(selected_ci).to_markdown(index=False) if selected_ci else "No bootstrap CI results."])
    atomic_text("\n".join(lines)+"\n",reports/"final_summary.md"); write_marker(reports/".PIPELINE_COMPLETE","14_summarize_pipeline",config,{},summary)
    if primary_success:
        write_marker(reports/".PRED_ROI_PRIMARY_SUCCESS","14_summarize_pipeline",config,{},summary); write_marker(reports/".FULL_AUTO_SUCCESS","14_summarize_pipeline",config,{},summary)
    else: atomic_json(summary,reports/".PIPELINE_COMPLETE_WITH_FAILED_GATES")
    finish({"primary_success":primary_success}); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
