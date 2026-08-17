#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from common import atomic_csv, atomic_json, atomic_text, load_config, sha256_file, tree_hash
from data import prepare_pair
from model_interface import build_model


FAMILIES = ["segresnet", "deeplabv3plus_resnet50_imagenet"]
KEY = ["patient_id", "series_uid", "phase"]


def load_result(cfg, fold, family):
    output = Path(cfg["output_root"]) / "development" / f"fold_{fold}" / family / "SUCCESS.json"
    report = Path(cfg["report_root"]) / "development" / f"fold_{fold}" / family
    if not output.is_file():
        raise FileNotFoundError(output)
    result = json.loads(output.read_text(encoding="utf-8"))
    predictions = pd.read_csv(report / "best_inner_valid_predictions.csv", dtype={"patient_id": str, "series_uid": str})
    return result, predictions


def cluster_bootstrap(paired: pd.DataFrame, draws: int, seed: int):
    cases = paired["series_uid"].drop_duplicates().to_numpy()
    groups = {case: paired.loc[paired["series_uid"].eq(case), "dice_gain"].to_numpy() for case in cases}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = cases[rng.integers(0, len(cases), size=len(cases))]
        estimates[draw] = np.concatenate([groups[case] for case in sampled]).mean()
    return {
        "mean": float(paired["dice_gain"].mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_gt_zero": float((estimates > 0).mean()),
        "resampling_unit": "series_uid with paired Pre/Post rows",
    }


def summarize(group):
    return {
        "n": int(len(group)),
        "segresnet_macro_dice": float(group["dice_segresnet"].mean()),
        "deeplab_macro_dice": float(group["dice_deeplab"].mean()),
        "deeplab_minus_segresnet": float(group["dice_gain"].mean()),
        "segresnet_failure_lt_02": float(group["dice_segresnet"].lt(0.2).mean()),
        "deeplab_failure_lt_02": float(group["dice_deeplab"].lt(0.2).mean()),
        "segresnet_failure_lt_05": float(group["dice_segresnet"].lt(0.5).mean()),
        "deeplab_failure_lt_05": float(group["dice_deeplab"].lt(0.5).mean()),
        "segresnet_zero_overlap": float(group["zero_overlap_segresnet"].mean()),
        "deeplab_zero_overlap": float(group["zero_overlap_deeplab"].mean()),
        "segresnet_area_ratio_mean": float(group["pred_gt_area_ratio_segresnet"].mean()),
        "deeplab_area_ratio_mean": float(group["pred_gt_area_ratio_deeplab"].mean()),
        "fraction_deeplab_improved": float(group["dice_gain"].gt(0).mean()),
    }


def paired_comparison(cfg, fold, seg_rows, deep_rows):
    selected_columns = KEY + [
        "dice", "iou", "intersection_pixels", "pred_pixels", "gt_pixels", "pred_gt_area_ratio",
        "zero_overlap", "image_path", "mask_path", "probability_mean_gt", "probability_mean_background",
        "probability_entropy_mean",
    ]
    paired = seg_rows[selected_columns].merge(
        deep_rows[selected_columns], on=KEY, how="inner", validate="one_to_one",
        suffixes=("_segresnet", "_deeplab"),
    )
    if len(paired) != len(seg_rows) or len(paired) != len(deep_rows):
        raise RuntimeError(f"Fold {fold}: prediction rows do not align")
    if not np.array_equal(paired["gt_pixels_segresnet"], paired["gt_pixels_deeplab"]):
        raise RuntimeError(f"Fold {fold}: GT pixels changed between models")
    paired["dice_gain"] = paired["dice_deeplab"] - paired["dice_segresnet"]
    paired["gt_size_quartile"] = pd.qcut(
        paired["gt_pixels_segresnet"], 4, labels=["Q1-smallest", "Q2", "Q3", "Q4-largest"]
    ).astype(str)
    paired["fold"] = int(fold)
    out = Path(cfg["report_root"]) / "development" / f"fold_{fold}" / "paired_comparison"
    atomic_csv(paired, out / "paired_phase_predictions.csv")
    rows = [{"stratum": "Overall", "level": "Overall", **summarize(paired)}]
    for column in ["phase", "gt_size_quartile"]:
        for level, group in paired.groupby(column, observed=True):
            rows.append({"stratum": column, "level": str(level), **summarize(group)})
    atomic_csv(pd.DataFrame(rows), out / "paired_stratified_summary.csv")
    bootstrap_cfg = cfg["bootstrap"]
    result = {
        "status": "success",
        "development_only": True,
        "outer_fold": int(fold),
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "overall": summarize(paired),
        "paired_cluster_bootstrap": cluster_bootstrap(
            paired, int(bootstrap_cfg["draws"]), int(bootstrap_cfg["seed"]) + int(fold)
        ),
    }
    atomic_json(result, out / "paired_comparison.json")
    return paired, result


def atomic_png(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"Failed to write {temporary}")
    os.replace(temporary, path)


def overlay_image(image, mask, probability, title):
    base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    pred = (probability >= 0.5).astype(np.uint8)
    gt_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(color, pred_contours, -1, (0, 0, 255), 2)
    heat = cv2.applyColorMap((np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), 0.45, heat, 0.55, 0)
    canvas = np.concatenate([cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), color, blended], axis=1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, title[:180], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def select_overlay_rows(rows):
    selections = []
    categories = [
        ("lowest_dice", rows.nsmallest(3, "dice")),
        ("highest_dice", rows.nlargest(2, "dice")),
        ("post_low", rows.loc[rows["phase"].eq("Post")].nsmallest(3, "dice")),
        ("small_lesion", rows.nsmallest(2, "gt_pixels")),
        ("oversegmentation", rows.nlargest(2, "pred_gt_area_ratio")),
        ("zero_overlap", rows.loc[rows["zero_overlap"]].head(3)),
    ]
    seen = set()
    for category, subset in categories:
        for _, row in subset.iterrows():
            key = (row["series_uid"], row["phase"])
            if key in seen:
                continue
            seen.add(key)
            selections.append((category, row))
    return selections


def generate_overlays(cfg, fold, family, result, rows, device):
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    model = build_model(family, cfg, load_pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output = Path(cfg["report_root"]) / "development" / f"fold_{fold}" / family / "overlays"
    manifest = []
    for category, row in select_overlay_rows(rows):
        image, mask, _ = prepare_pair(row["image_path"], row["mask_path"], cfg)
        x = torch.from_numpy(image[None, None]).float().to(device)
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
        title = (
            f"{family} fold={fold} {row['phase']} {row['series_uid']} "
            f"Dice={row['dice']:.4f} area={row['pred_gt_area_ratio']:.3f} category={category}"
        )
        filename = f"{category}_{row['series_uid'].replace('/', '_')}_{row['phase']}.png"
        atomic_png(overlay_image(image, mask, probability, title), output / filename)
        manifest.append({"category": category, "series_uid": row["series_uid"], "phase": row["phase"], "path": str(output / filename)})
    atomic_csv(pd.DataFrame(manifest), output / "overlay_manifest.csv")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def verify_protected_assets(cfg):
    baseline_path = Path(cfg["report_root"]) / "00_preflight/protected_assets_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    checks = {}
    for path, record in baseline.items():
        current = sha256_file(path) if Path(path).is_file() else tree_hash(path)
        checks[path] = {"baseline": record["tree_sha256"], "current": current, "unchanged": current == record["tree_sha256"]}
        if current != record["tree_sha256"]:
            raise RuntimeError(f"Protected v5/fusion asset changed: {path}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    results = {}
    paired_results = {}
    all_metrics = []
    for fold in [int(value) for value in cfg["development"]["outer_folds"]]:
        seg_result, seg_rows = load_result(cfg, fold, "segresnet")
        deep_result, deep_rows = load_result(cfg, fold, "deeplabv3plus_resnet50_imagenet")
        paired, comparison = paired_comparison(cfg, fold, seg_rows, deep_rows)
        paired_results[str(fold)] = comparison
        results[str(fold)] = {"segresnet": seg_result, "deeplabv3plus_resnet50_imagenet": deep_result}
        for family, result, rows in [
            ("segresnet", seg_result, seg_rows),
            ("deeplabv3plus_resnet50_imagenet", deep_result, deep_rows),
        ]:
            generate_overlays(cfg, fold, family, result, rows, device)
            all_metrics.append({"outer_fold": fold, "family": family, **result["best_inner_valid_metrics"]})

    metrics = pd.DataFrame(all_metrics)
    atomic_csv(metrics, Path(cfg["report_root"]) / "DEVELOPMENT_METRICS.csv")
    means = metrics.groupby("family", as_index=True)[["macro_dice", "post_dice", "failure_lt_02"]].mean()
    seg = means.loc["segresnet"]
    deep = means.loc["deeplabv3plus_resnet50_imagenet"]
    fold_deltas = {}
    for fold in [1, 4]:
        fold_frame = metrics.loc[metrics["outer_fold"].eq(fold)].set_index("family")
        fold_deltas[str(fold)] = float(
            fold_frame.loc["deeplabv3plus_resnet50_imagenet", "macro_dice"]
            - fold_frame.loc["segresnet", "macro_dice"]
        )
    rule = cfg["promotion_rule"]
    mean_gain = float(deep["macro_dice"] - seg["macro_dice"])
    post_change = float(deep["post_dice"] - seg["post_dice"])
    severe_change = float(deep["failure_lt_02"] - seg["failure_lt_02"])
    checks = {
        "minimum_mean_gain_met": mean_gain >= float(rule["deeplab_minimum_mean_gain"]),
        "nonnegative_macro_gain_each_split": all(value >= 0 for value in fold_deltas.values()),
        "post_not_materially_worse": post_change >= -float(rule["maximum_post_dice_deterioration"]),
        "severe_tail_not_materially_worse": severe_change <= float(rule["maximum_severe_failure_rate_deterioration"]),
    }
    deep_promoted = all(checks.values())
    primary = "deeplabv3plus_resnet50_imagenet" if deep_promoted else str(rule["fallback_primary"])
    comparator = "segresnet" if deep_promoted else "deeplabv3plus_resnet50_imagenet"
    protected = verify_protected_assets(cfg)
    historical = json.loads(Path(cfg["sources"]["pilot_fold1_metrics"]).read_text(encoding="utf-8"))
    v6_fold1_seg = results["1"]["segresnet"]["best_inner_valid_metrics"]["macro_dice"]
    decision = {
        "status": "frozen_development_promotion_decision",
        "created_before_full_strict": True,
        "full_strict_started": False,
        "development_only": True,
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "primary": primary,
        "comparator": comparator,
        "promotion_rule": rule,
        "rule_checks": checks,
        "aggregate": {
            "segresnet_mean_macro_dice": float(seg["macro_dice"]),
            "deeplab_mean_macro_dice": float(deep["macro_dice"]),
            "deeplab_minus_segresnet_mean_macro_dice": mean_gain,
            "segresnet_mean_post_dice": float(seg["post_dice"]),
            "deeplab_mean_post_dice": float(deep["post_dice"]),
            "deeplab_minus_segresnet_post_dice": post_change,
            "segresnet_mean_failure_lt_02": float(seg["failure_lt_02"]),
            "deeplab_mean_failure_lt_02": float(deep["failure_lt_02"]),
            "deeplab_minus_segresnet_failure_lt_02": severe_change,
            "fold_macro_dice_deltas": fold_deltas,
        },
        "fold_results": results,
        "paired_comparisons": paired_results,
        "segresnet_fold1_migration_reproduction": {
            "historical_macro_dice": float(historical["best_inner_valid_metrics"]["macro_dice"]),
            "v6_macro_dice": float(v6_fold1_seg),
            "difference": float(v6_fold1_seg - historical["best_inner_valid_metrics"]["macro_dice"]),
        },
        "frozen_recipe": {
            "config_path": cfg["_config_path"],
            "config_sha256": cfg["_config_sha256"],
            "threshold": 0.5,
            "augmentation": cfg["augmentation"],
            "loss": cfg["loss"],
            "models": cfg["models"],
            "pretrained": cfg["pretrained"],
        },
        "protected_assets_unchanged": protected,
    }
    decision_path = Path(cfg["report_root"]) / "PROMOTION_DECISION.json"
    atomic_json(decision, decision_path)
    lines = [
        "# V6 development-only promotion decision",
        "",
        f"Primary: `{primary}`",
        f"Comparator: `{comparator}`",
        "",
        f"Mean macro Dice: SegResNet {seg['macro_dice']:.4f}; DeepLabV3+ {deep['macro_dice']:.4f}; delta {mean_gain:+.4f}.",
        f"Mean Post Dice delta (DeepLab-SegResNet): {post_change:+.4f}.",
        f"Mean severe-failure-rate delta: {severe_change:+.4f}.",
        f"Fold macro deltas: fold-1 {fold_deltas['1']:+.4f}; fold-4 {fold_deltas['4']:+.4f}.",
        "",
        "No outer holdout or independent Valid cohort was evaluated or used for this decision.",
    ]
    atomic_text("\n".join(lines) + "\n", Path(cfg["report_root"]) / "DEVELOPMENT_DECISION.md")
    atomic_json({"status": "success", "promotion_decision": str(decision_path), "primary": primary, "comparator": comparator}, Path(cfg["report_root"]) / "SUCCESS.json")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
