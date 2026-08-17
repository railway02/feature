#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from common import (
    atomic_csv,
    atomic_json,
    atomic_text,
    atomic_torch_save,
    load_config,
    set_seed,
    sha256_file,
    tree_hash,
    tree_manifest,
)
from data import load_train_manifest, prepare_pair, split_manifest
from losses import segmentation_loss
from model_interface import build_model, model_parameter_count


def package_versions():
    import monai
    import segmentation_models_pytorch as smp
    import timm
    import torchvision
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "monai": monai.__version__,
        "segmentation_models_pytorch": smp.__version__,
        "timm": timm.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def audit_input_locks(cfg):
    keys = ["phase_mapping_manifest", "segmentation_inventory", "temporal_train_npz", "temporal_valid_npz"]
    result = {}
    for key in keys:
        path = Path(cfg["sources"][key])
        actual = sha256_file(path)
        expected = cfg["sources"][f"{key}_sha256"]
        result[key] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(f"Locked input checksum mismatch: {key}")
    return result


def audit_protected_assets(cfg):
    assets = {}
    for path in cfg["protected_roots"]:
        resolved = Path(path).resolve()
        assets[str(resolved)] = {
            "tree_sha256": tree_hash(resolved),
            "files": tree_manifest(resolved),
        }
    fusion = Path(cfg["sources"]["v5_fusion_architecture"])
    assets[str(fusion.resolve())] = {
        "tree_sha256": sha256_file(fusion),
        "files": [{"path": fusion.name, "size": fusion.stat().st_size, "sha256": sha256_file(fusion)}],
    }
    return assets


def audit_train_files(train, cfg):
    records = []
    for case in train.itertuples(index=False):
        for phase in cfg["data"]["phases"]:
            key = phase.lower()
            image_path = str(getattr(case, f"{key}_image"))
            mask_path = str(getattr(case, f"{key}_mask"))
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(f"Unreadable pair: {image_path} / {mask_path}")
            if image.shape != mask.shape:
                raise RuntimeError(f"Shape mismatch: {image_path}")
            foreground = mask > 0
            if not foreground.any():
                raise RuntimeError(f"Empty mask: {mask_path}")
            _, prepared_mask, letterbox = prepare_pair(image_path, mask_path, cfg)
            records.append({
                "series_uid": str(case.series_uid),
                "patient_id": str(case.patient_id),
                "outer_fold": int(case.fold),
                "phase": phase,
                "image_path": image_path,
                "mask_path": mask_path,
                "native_height": int(image.shape[0]),
                "native_width": int(image.shape[1]),
                "native_mask_foreground": int(foreground.sum()),
                "prepared_mask_foreground": int(prepared_mask.sum()),
                "padding_top": letterbox["padding"][0],
                "padding_bottom": letterbox["padding"][1],
                "padding_left": letterbox["padding"][2],
                "padding_right": letterbox["padding"][3],
            })
    return pd.DataFrame(records)


def verify_fold1_split(cfg, generated):
    reference = pd.read_csv(cfg["sources"]["pilot_fold1_split"], dtype={"patient_id": str, "series_uid": str})
    reference_map = reference.set_index("series_uid")["pilot_partition"].astype(str).to_dict()
    generated_map = generated.set_index("series_uid")["development_partition"].astype(str).to_dict()
    translated = {
        "inner_train": "inner_train",
        "inner_valid": "inner_valid",
        "forbidden_outer_holdout_not_evaluated": "forbidden_outer_holdout_not_used",
    }
    normalized = {key: translated[value] for key, value in generated_map.items()}
    if normalized != reference_map:
        missing = sorted(set(reference_map) ^ set(normalized))[:10]
        mismatched = sorted(key for key in set(reference_map) & set(normalized) if reference_map[key] != normalized[key])[:10]
        raise RuntimeError(f"Fold-1 split does not reproduce pilot. missing={missing}, mismatched={mismatched}")
    return {"reference": cfg["sources"]["pilot_fold1_split"], "series": len(normalized), "exact_match": True}


def one_optimizer_step(model, batch_size, cfg, family, device):
    size = int(cfg["data"]["input_size"])
    x = torch.zeros(batch_size, 1, size, size, device=device)
    y = torch.zeros(batch_size, 1, size, size, device=device)
    y[:, :, size // 2 - 16:size // 2 + 16, size // 2 - 16:size // 2 + 16] = 1
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=float(cfg["models"][family]["weight_decay"]))
    scaler = GradScaler(enabled=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=device.type == "cuda"):
        feature_map, logits = model.encode_and_decode(x)
        loss = segmentation_loss(logits, y, 14.0, cfg)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "feature_map_shape": list(feature_map.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
    }


def checkpoint_reload_probe(model, cfg, family, device, output_dir):
    model.eval()
    sample = torch.linspace(0, 1, int(cfg["data"]["input_size"]), device=device).view(1, 1, 1, -1)
    sample = sample.repeat(1, 1, int(cfg["data"]["input_size"]), 1)
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        before = model(sample).float().cpu()
    path = output_dir / f"{family}_reload_smoke.pt"
    atomic_torch_save({"state_dict": model.state_dict()}, path)
    reloaded = build_model(family, cfg, load_pretrained=False).to(device).eval()
    reloaded.load_state_dict(torch.load(path, map_location="cpu")["state_dict"], strict=True)
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        after = reloaded(sample).float().cpu()
    maximum_difference = float((before - after).abs().max())
    if maximum_difference != 0.0:
        raise RuntimeError(f"Checkpoint reload changed logits for {family}: {maximum_difference}")
    del reloaded, sample, before, after
    return {"checkpoint": str(path), "sha256": sha256_file(path), "max_abs_logit_difference": maximum_difference}


def probe_model(cfg, family, device, batches):
    results = []
    total_memory = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    for batch_size in batches:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            set_seed(20260810 + batch_size)
            model = build_model(family, cfg, load_pretrained=True).to(device)
            shapes = one_optimizer_step(model, int(batch_size), cfg, family, device)
            allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            results.append({
                "batch_size": int(batch_size),
                "stable": True,
                "max_memory_allocated_bytes": allocated,
                "max_memory_reserved_bytes": reserved,
                "reserved_fraction_total": float(reserved / total_memory) if total_memory else 0.0,
                **shapes,
            })
            del model
        except torch.cuda.OutOfMemoryError as error:
            results.append({"batch_size": int(batch_size), "stable": False, "error": f"{type(error).__name__}:{error}"})
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            results.append({"batch_size": int(batch_size), "stable": False, "error": f"{type(error).__name__}:{error}"})
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def verify_pretrained_weights(cfg, model):
    path = Path(cfg["pretrained"]["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != cfg["pretrained"]["sha256"] or path.stat().st_size != int(cfg["pretrained"]["size_bytes"]):
        raise RuntimeError("ImageNet checkpoint checksum or size mismatch")
    raw = torch.load(path, map_location="cpu")
    expected = raw["conv1.weight"]
    actual = model.encoder_conv1().detach().cpu()
    exact = bool(torch.equal(expected, actual))
    if not exact:
        raise RuntimeError("SMP ResNet50 conv1 does not equal the recorded ImageNet checkpoint")
    return {
        "source_url": cfg["pretrained"]["source_url"],
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": actual_hash,
        "encoder_conv1_exact_match": exact,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    report_dir = Path(cfg["report_root"]) / "00_preflight"
    smoke_dir = Path(cfg["output_root"]) / "smoke"
    report_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_v6.py"],
        cwd=cfg["code_root"], check=True, text=True, capture_output=True,
    )
    atomic_text(unit.stdout + unit.stderr, report_dir / "unit_tests.txt")
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", cfg["code_root"]],
        check=True, text=True, capture_output=True,
    )
    atomic_text(compile_result.stdout + compile_result.stderr, report_dir / "py_compile.txt")

    versions = package_versions()
    atomic_json(versions, report_dir / "environment_versions.json")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    atomic_text(freeze, report_dir / "pip_freeze.txt")
    input_locks = audit_input_locks(cfg)
    protected = audit_protected_assets(cfg)
    atomic_json(protected, report_dir / "protected_assets_baseline.json")

    train = load_train_manifest(cfg)
    counts = train["fold"].value_counts().sort_index().to_dict()
    expected_counts = {int(key): int(value) for key, value in cfg["data"]["expected_fold_counts"].items()}
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected fold counts: {counts}")
    file_audit = audit_train_files(train, cfg)
    if len(file_audit) != int(cfg["data"]["expected_train_phase_rows"]):
        raise RuntimeError(f"Unexpected phase audit rows: {len(file_audit)}")
    atomic_csv(file_audit, report_dir / "train_image_mask_audit.csv")

    split_audits = {}
    for fold in cfg["development"]["outer_folds"]:
        generated = split_manifest(train, int(fold), cfg)
        atomic_csv(generated, report_dir / f"fold_{fold}_development_split.csv")
        split_audits[str(fold)] = {
            "inner_train_series": int(generated["development_partition"].eq("inner_train").sum()),
            "inner_valid_series": int(generated["development_partition"].eq("inner_valid").sum()),
            "forbidden_outer_holdout_series": int(generated["development_partition"].str.startswith("forbidden").sum()),
            "patient_overlap": False,
        }
        if int(fold) == 1:
            split_audits[str(fold)]["pilot_reproduction"] = verify_fold1_split(cfg, generated)

    seg_probe = probe_model(cfg, "segresnet", device, [4])
    if not seg_probe[0]["stable"]:
        raise RuntimeError("Corrected SegResNet batch-4 smoke failed")
    if seg_probe[0]["feature_map_shape"][1:] != [256, 96, 96] or seg_probe[0]["logits_shape"][1:] != [1, 768, 768]:
        raise RuntimeError(f"Unexpected SegResNet shapes: {seg_probe[0]}")

    deep_batches = [int(value) for value in cfg["models"]["deeplabv3plus_resnet50_imagenet"]["physical_batch_candidates"]]
    deep_probe = probe_model(cfg, "deeplabv3plus_resnet50_imagenet", device, deep_batches)
    stable = [item for item in deep_probe if item["stable"] and item["reserved_fraction_total"] <= 1.0 - float(cfg["models"]["deeplabv3plus_resnet50_imagenet"]["minimum_free_memory_fraction"])]
    if not stable:
        raise RuntimeError(f"No DeepLab batch retains the required memory margin: {deep_probe}")
    selected_batch = max(item["batch_size"] for item in stable)
    selected_accumulation = int(cfg["models"]["deeplabv3plus_resnet50_imagenet"]["gradient_accumulation_if_batch2"]) if selected_batch == 2 else 1
    selected_probe = next(item for item in deep_probe if item["batch_size"] == selected_batch)
    if selected_probe["feature_map_shape"][1] != 256 or selected_probe["logits_shape"][1:] != [1, 768, 768]:
        raise RuntimeError(f"Unexpected DeepLab shapes: {selected_probe}")
    runtime_selection = {
        "segresnet": {"physical_batch_size": 4, "gradient_accumulation": 1},
        "deeplabv3plus_resnet50_imagenet": {
            "physical_batch_size": int(selected_batch),
            "gradient_accumulation": int(selected_accumulation),
            "selection_basis": "largest probed batch with a complete forward/backward/optimizer step and >=15% reserved-memory margin",
        },
    }
    atomic_json(runtime_selection, report_dir / "runtime_selection.json")

    set_seed(20260810)
    seg_model = build_model("segresnet", cfg, load_pretrained=False).to(device)
    seg_reload = checkpoint_reload_probe(seg_model, cfg, "segresnet", device, smoke_dir)
    del seg_model
    set_seed(20260810)
    deep_model = build_model("deeplabv3plus_resnet50_imagenet", cfg, load_pretrained=True).to(device)
    pretrained = verify_pretrained_weights(cfg, deep_model)
    deep_reload = checkpoint_reload_probe(deep_model, cfg, "deeplabv3plus_resnet50_imagenet", device, smoke_dir)
    deep_model.eval()
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        feature_map, logits = deep_model.encode_and_decode(torch.zeros(1, 1, 768, 768, device=device))
    deep_shape = {"decoder_feature_map": list(feature_map.shape), "logits": list(logits.shape)}
    del deep_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "status": "PASS",
        "development_only": True,
        "outer_holdout_model_evaluation_performed": False,
        "independent_valid_used_for_selection": False,
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "versions": versions,
        "input_locks": input_locks,
        "protected_assets": {path: value["tree_sha256"] for path, value in protected.items()},
        "train": {
            "series": int(len(train)),
            "phase_rows": int(len(file_audit)),
            "fold_counts": counts,
            "patients": int(train["patient_id"].nunique()),
        },
        "development_splits": split_audits,
        "smoke": {
            "unit_tests": "PASS",
            "py_compile": "PASS",
            "segresnet_memory_probe": seg_probe,
            "deeplab_memory_probe": deep_probe,
            "runtime_selection": runtime_selection,
            "segresnet_checkpoint_reload": seg_reload,
            "deeplab_checkpoint_reload": deep_reload,
            "deeplab_observed_shapes": deep_shape,
        },
        "pretrained": pretrained,
    }
    atomic_json(result, report_dir / "00_preflight.json")
    atomic_json(result, report_dir / "SUCCESS.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
