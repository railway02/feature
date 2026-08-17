#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path

from common import configure_runtime, load_config, require_file, sha256_file, stage_logger, write_marker, atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    finish = stage_logger("00_static_check")
    required = [
        "train_excel", "valid_excel", "train_manifest", "valid_manifest", "checkpoint",
        "fixed_trainer", "cave_python", "prediction_python", "base_cave_config",
        "v3_extractor", "v3_base_config", "v3_override_config",
    ]
    evidence = {}
    for key in required:
        expected = config.get(f"{key}_sha256")
        if key == "checkpoint": expected = config["checkpoint_sha256"]
        if key == "fixed_trainer": expected = config["fixed_trainer_sha256"]
        evidence[key] = require_file(config[key], expected)
    for key in ("project_root", "raw_root", "updated_root", "cave_code_root", "cave_repo"):
        path = Path(config[key]).resolve()
        if not path.is_dir(): raise FileNotFoundError(path)
        evidence[key] = {"path": str(path)}
    code_root=Path(config["paths"]["code"])
    required_scripts=[f"{index:02d}_{name}.py" for index,name in enumerate(["static_check","scan_current_assets","build_authoritative_roi_manifest","infer_reference_rule_and_alignment","build_segmentation_dataset","train_segmentation_oof","infer_train_oof_valid_masks","build_roi_manifests","extract_mask_morphology","extract_roi_cave_featurebank","build_roi_cave_tables","build_adverse_tasks","train_adverse_models_fixed","run_ablations","summarize_pipeline"])]
    required_scripts += ["common.py","assets.py","segmentation.py","roi.py","build_patient_aggregations.py","compact_cave_featurebank.py","resume_guard.py","test_synthetic.py"]
    for name in required_scripts:
        item=code_root/name; require_file(item); evidence[f"code:{name}"]={"path":str(item.resolve()),"size_bytes":item.stat().st_size,"sha256":sha256_file(item)}
    disk=shutil.disk_usage(config["project_root"]); evidence["disk"]={"total_bytes":disk.total,"used_bytes":disk.used,"free_bytes":disk.free,"compacted_three_branch_strategy":True}
    if disk.free<80*1024**3:
        raise RuntimeError(f"Insufficient free disk for sequential compacted ROI-CAVE extraction: {disk.free/1024**3:.1f} GiB")
    checks = [
        (config["cave_python"], "import torch,cv2,nibabel,numpy,pandas; assert torch.cuda.is_available(); print(torch.__version__)"),
        (config["prediction_python"], "import torch,sklearn,numpy,pandas,joblib; from sklearn.model_selection import StratifiedGroupKFold; print(sklearn.__version__)"),
    ]
    env = os.environ.copy()
    for python, code in checks:
        result = subprocess.run([python, "-c", code], env=env, check=True, text=True, capture_output=True)
        evidence[f"environment:{python}"] = result.stdout.strip()
    train_sha = sha256_file(Path(config["train_manifest"]))
    valid_sha = sha256_file(Path(config["valid_manifest"]))
    evidence["manifest_hashes"] = {"train": train_sha, "valid": valid_sha}
    report_root = Path(config["paths"]["reports"])
    atomic_json(evidence, report_root / "static_check.json")
    write_marker(report_root / ".STATIC_SUCCESS", "00_static_check", config, {}, evidence)
    finish({"checks": len(evidence)})
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
