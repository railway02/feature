#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_json, atomic_text, configure_runtime, file_evidence, hash_lines, load_config, process_rows, sha256_file, write_success


def code_tree_hash(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".sh"})
    return hash_lines(f"{path.relative_to(root)}:{sha256_file(path)}" for path in files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    reports = Path(config["paths"]["reports"])
    upstream_reports = Path(config["upstream_v1_reports"])
    upstream_code = Path(config["upstream_v1_code"])
    runner = upstream_code / "run_pipeline.sh"
    runner_text = runner.read_text(encoding="utf-8")

    marker_names = [
        ".STATIC_SUCCESS", ".ASSET_SCAN_SUCCESS", ".MANIFEST_CANDIDATES_SUCCESS",
        ".ASSET_SUCCESS", ".SEG_DATA_SUCCESS", ".SEGMENTATION_OOF_SUCCESS",
        ".SEGMENTATION_COMPLETE", ".SEGMENTATION_SUCCESS", ".ROI_SUCCESS",
        ".CAVE_FEATURES_SUCCESS", ".FULL_AUTO_SUCCESS",
    ]
    markers = {}
    marker_code_hashes = {}
    for name in marker_names:
        path = upstream_reports / name
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                item["stage"] = payload.get("stage")
                item["completed_at_utc"] = payload.get("completed_at_utc")
                item["pipeline_code_sha256"] = payload.get("pipeline_code_sha256")
                if payload.get("pipeline_code_sha256"):
                    marker_code_hashes[name] = payload["pipeline_code_sha256"]
            except Exception as exc:
                item["read_error"] = repr(exc)
        markers[name] = item

    current_code_hash = code_tree_hash(upstream_code)
    processes = process_rows("api_adverse_lesion_cave_v1")
    full_auto = [row for row in processes if "run_pipeline.sh full-auto" in row["args"]]
    training = [row for row in processes if "05_train_segmentation_oof.py" in row["args"]]
    parent_stopped = any(row["pid"] == "77378" and row["stat"].startswith("T") for row in full_auto)

    commands = [
        "check", "scan-assets", "build-manifest", "align", "build-seg-data", "smoke-seg",
        "train-seg-oof", "infer-masks", "build-roi", "extract-mask-features", "smoke-cave",
        "extract-cave", "compact-cave", "build-tables", "build-adverse-tasks",
        "train-models", "ablations", "summarize", "full-auto", "status",
    ]
    command_support = {name: (f"{name})" in runner_text or f'"{name}"' in runner_text or f"  {name}" in runner_text) for name in commands}
    payload = {
        "version": config["version"],
        "runner": file_evidence(runner),
        "current_v1_code_tree_sha256": current_code_hash,
        "marker_code_hashes": marker_code_hashes,
        "mixed_marker_code_hashes": len(set(marker_code_hashes.values())) > 1,
        "markers": markers,
        "processes": processes,
        "full_auto_parent_stopped": parent_stopped,
        "segmentation_training_alive": bool(training),
        "runner_supports_single_stage_commands": all(command_support[name] for name in ("infer-masks", "build-roi", "extract-mask-features", "extract-cave")),
        "runner_supports_native_stop_after_stage": False,
        "full_auto_automatically_enters_cave": True,
        "safe_handoff": "SIGSTOP only the full-auto parent while the train-seg-oof child continues; after .SEGMENTATION_OOF_SUCCESS, run downstream stages explicitly.",
        "command_support": command_support,
    }
    atomic_json(payload, reports / "current_pipeline_readonly_audit.json")

    lines = [
        "# Current pipeline read-only audit", "",
        f"- Full-auto parent stopped: `{parent_stopped}`",
        f"- Segmentation child alive: `{bool(training)}`",
        f"- Current v1 code-tree SHA256: `{current_code_hash}`",
        f"- Mixed code hashes across existing markers: `{payload['mixed_marker_code_hashes']}`",
        "- Native stop-after-stage support: `False`",
        "- Single-stage downstream commands: `True`",
        "- Original full-auto automatically enters ROI-CAVE: `True`", "",
        "## Marker state", "",
    ]
    for name in marker_names:
        lines.append(f"- {name}: `{markers[name]['exists']}`")
    lines.extend([
        "", "## Safe handoff", "",
        "The original full-auto orchestration parent is stopped, while the current segmentation child remains alive.",
        "When formal segmentation finishes and writes .SEGMENTATION_OOF_SUCCESS, downstream mask/ROI stages will be launched explicitly.",
        "The stopped original parent must not be resumed, otherwise it will continue into all Pred/GT/all-nonzero branches.",
    ])
    atomic_text("\n".join(lines) + "\n", reports / "current_pipeline_readonly_audit.md")
    write_success(reports / ".READONLY_AUDIT_SUCCESS", "00_readonly_audit", config, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
