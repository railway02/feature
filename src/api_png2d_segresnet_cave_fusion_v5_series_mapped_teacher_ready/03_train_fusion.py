#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from common import (
    atomic_csv,
    atomic_json,
    brier,
    load_config,
    load_temporal,
    load_train_folds,
    patient_inner_split,
    resolve_path,
    safe_ap,
    safe_auc,
    set_seed,
)
from fusion_models import OutcomeModel


def spatial_model_name(strategy, fold):
    strategy = strategy.casefold()
    if strategy == "pilot_single":
        return "pilot"
    if strategy == "external_checkpoint":
        return "external"
    if strategy == "strict_crossfit":
        return f"fold_{fold}"
    raise ValueError(strategy)


def spatial_key(cfg):
    representation = str(cfg["spatial"].get("representation", "global_only")).casefold()
    keys = {
        "global_only": "global",
        "global_gt_roi": "gt_combined",
        "global_pred_roi": "pred_combined",
    }
    if representation not in keys:
        raise ValueError("spatial.representation must be global_only, global_gt_roi, or global_pred_roi")
    return keys[representation]


def load_spatial(out, model_name, split, key, task_uid):
    path = out / "seg_features" / model_name / f"{split.casefold()}.npz"
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z[key], dtype=np.float32)
        uid = z["series_uid"].astype(str)

    if not np.array_equal(uid, task_uid.astype(str)):
        raise AssertionError(f"{path}: UID order mismatch")
    if not np.isfinite(x).all():
        raise AssertionError(f"{path}: spatial feature has NaN/Inf")
    return x


def spatial_feature_root(cfg, out):
    configured = str(cfg["spatial"].get("feature_root", "") or "").strip()
    return resolve_path(configured, cfg["project_root"]) if configured else out


def clean_scalar_foldwise(train_scalar, valid_scalar, development, cfg):
    """Train-only scalar cleanup without PCA or feature engineering."""
    sc = cfg["temporal"]["scalar_cleaning"]
    x = np.asarray(train_scalar, dtype=np.float64)
    xv = np.asarray(valid_scalar, dtype=np.float64)
    dev_x = x[development]
    finite = np.isfinite(dev_x)
    missing_rate = 1.0 - finite.mean(axis=0)
    medians = np.full(dev_x.shape[1], np.nan, dtype=np.float64)
    for col in np.flatnonzero(missing_rate <= float(sc["max_missing_rate"])):
        values = dev_x[finite[:, col], col]
        if values.size:
            medians[col] = np.median(values)

    keep_missing = np.flatnonzero(
        (missing_rate <= float(sc["max_missing_rate"])) & np.isfinite(medians)
    )
    if keep_missing.size == 0:
        raise RuntimeError("Scalar cleaning removed every column by missingness")

    selected_dev = dev_x[:, keep_missing]
    lower = np.nanquantile(selected_dev, float(sc["lower_quantile"]), axis=0)
    upper = np.nanquantile(selected_dev, float(sc["upper_quantile"]), axis=0)
    selected_medians = medians[keep_missing]
    dev_clean = np.clip(selected_dev, lower, upper)
    dev_clean = np.where(np.isfinite(dev_clean), dev_clean, selected_medians[None, :])
    keep_variance = np.flatnonzero(
        np.var(dev_clean, axis=0) > float(sc["min_variance"])
    )
    if keep_variance.size == 0:
        raise RuntimeError("Scalar cleaning removed every column by variance")

    lower = lower[keep_variance]
    upper = upper[keep_variance]
    selected_medians = selected_medians[keep_variance]
    columns = keep_missing[keep_variance]

    def transform(raw):
        values = np.asarray(raw, dtype=np.float64)[:, columns]
        values = np.clip(values, lower, upper)
        values = np.where(np.isfinite(values), values, selected_medians[None, :])
        return values

    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    train_clean = scaler.fit_transform(transform(x[development]))
    full_train = scaler.transform(transform(x)).astype(np.float32)
    full_valid = scaler.transform(transform(xv)).astype(np.float32)
    if not (np.isfinite(train_clean).all() and np.isfinite(full_train).all() and np.isfinite(full_valid).all()):
        raise RuntimeError("Nonfinite scalar values remain after fold-wise cleaning")

    return full_train, full_valid, {
        "policy": "outer_development_only_missing_filter_clip_median_robust_scale_no_pca",
        "input_columns": int(x.shape[1]),
        "kept_after_missing": int(keep_missing.size),
        "kept_after_variance": int(columns.size),
        "max_missing_rate": float(sc["max_missing_rate"]),
        "lower_quantile": float(sc["lower_quantile"]),
        "upper_quantile": float(sc["upper_quantile"]),
        "min_variance": float(sc["min_variance"]),
    }


def make_loader(spatial, temporal, y, idx, batch, shuffle, seed):
    idx = np.asarray(idx, dtype=int)

    s = (
        torch.from_numpy(spatial[idx]).float()
        if spatial is not None
        else torch.empty((len(idx), 0))
    )
    t = (
        torch.from_numpy(temporal[idx]).float()
        if temporal is not None
        else torch.empty((len(idx), 0))
    )
    yy = torch.from_numpy(y[idx].astype(np.float32)).view(-1, 1)

    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(s, t, yy),
        batch_size=batch,
        shuffle=shuffle,
        generator=g,
        drop_last=False,
    )


def build_model(mode, spatial, temporal, cfg, device):
    return OutcomeModel(
        mode=mode,
        spatial_dim=spatial.shape[1] if spatial is not None else 1,
        temporal_dim=temporal.shape[1] if temporal is not None else 1,
        hidden_dim=int(cfg["fusion"]["hidden_dim"]),
        fusion_mid_dim=int(cfg["fusion"]["fusion_mid_dim"]),
        dropout=float(cfg["fusion"]["dropout"]),
    ).to(device)


def train_epoch(model, dl, optimizer, scaler, device, mode, pos_weight, amp):
    model.train()
    losses = []
    pw = torch.tensor([pos_weight], device=device)

    for spatial, temporal, y in dl:
        spatial = spatial.to(device)
        temporal = temporal.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            out = model(
                None if mode == "cave_only" else spatial,
                None if mode == "spatial_only" else temporal,
            )
            loss = F.binary_cross_entropy_with_logits(
                out["logit"],
                y,
                pos_weight=pw,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


@torch.no_grad()
def predict(model, spatial, temporal, idx, cfg, device, mode, collect_gates=False):
    n = len(spatial) if spatial is not None else len(temporal)
    fake = np.zeros(n, dtype=int)
    probs = []
    gate2 = []
    gatet = []

    dl = make_loader(
        spatial,
        temporal,
        fake,
        idx,
        int(cfg["fusion"]["batch_size"]),
        False,
        0,
    )

    model.eval()
    for s, t, _ in dl:
        s = s.to(device)
        t = t.to(device)

        out = model(
            None if mode == "cave_only" else s,
            None if mode == "spatial_only" else t,
        )
        probs.append(torch.sigmoid(out["logit"]).cpu().numpy().ravel())

        if collect_gates and "gate_2d" in out:
            gate2.append(out["gate_2d"].cpu().numpy())
            gatet.append(out["gate_t"].cpu().numpy())

    result = {"prob": np.concatenate(probs)}
    if gate2:
        result["gate_2d"] = np.concatenate(gate2)
        result["gate_t"] = np.concatenate(gatet)
    return result


def select_epoch(mode, spatial, temporal, y, tr, va, cfg, device, seed, fold_dir):
    set_seed(seed)
    fc = cfg["fusion"]

    model = build_model(mode, spatial, temporal, cfg, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(fc["learning_rate"]),
        weight_decay=float(fc["weight_decay"]),
    )

    amp = bool(fc["amp"] and device.type == "cuda")
    scaler = GradScaler(enabled=amp)

    pos = max(1, int(y[tr].sum()))
    neg = max(1, len(tr) - pos)
    pos_weight = neg / pos

    dl = make_loader(
        spatial,
        temporal,
        y,
        tr,
        int(fc["batch_size"]),
        True,
        seed,
    )

    best_ap = -1.0
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, int(fc["max_epochs"]) + 1):
        loss = train_epoch(
            model,
            dl,
            optimizer,
            scaler,
            device,
            mode,
            pos_weight,
            amp,
        )

        p = predict(model, spatial, temporal, va, cfg, device, mode)["prob"]
        ap = safe_ap(y[va], p)
        auc = safe_auc(y[va], p)

        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "valid_AUPRC": ap,
            "valid_AUROC": auc,
        })

        print(
            f"[fusion:{mode}] epoch={epoch:03d} "
            f"loss={loss:.5f} val_AUPRC={ap:.5f}",
            flush=True,
        )

        if ap > best_ap + 1e-6:
            best_ap = ap
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if epoch >= int(fc["min_epochs"]) and bad >= int(fc["patience"]):
            break

    pd.DataFrame(history).to_csv(fold_dir / "epoch_search.csv", index=False)
    return best_epoch, best_ap


def refit(mode, spatial, temporal, y, idx, epochs, cfg, device, seed, fold_dir):
    set_seed(seed)
    fc = cfg["fusion"]

    model = build_model(mode, spatial, temporal, cfg, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(fc["learning_rate"]),
        weight_decay=float(fc["weight_decay"]),
    )

    amp = bool(fc["amp"] and device.type == "cuda")
    scaler = GradScaler(enabled=amp)

    pos = max(1, int(y[idx].sum()))
    neg = max(1, len(idx) - pos)
    pos_weight = neg / pos

    dl = make_loader(
        spatial,
        temporal,
        y,
        idx,
        int(fc["batch_size"]),
        True,
        seed,
    )

    for _ in range(epochs):
        train_epoch(
            model,
            dl,
            optimizer,
            scaler,
            device,
            mode,
            pos_weight,
            amp,
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mode": mode,
            "epochs": int(epochs),
        },
        fold_dir / "model.pt",
    )
    return model


def run_mode(mode, cfg, out, device):
    tr = load_temporal(cfg, "Train")
    va = load_temporal(cfg, "Valid")
    folds = load_train_folds(cfg, tr)

    y = tr["target"]
    yv = va["target"]
    groups = tr["patient_id"]

    use_scalar = bool(cfg.get("temporal", {}).get("include_scalar", False))
    if use_scalar and ("scalar" not in tr or "scalar" not in va):
        raise KeyError("temporal.include_scalar requires scalar arrays in both CAVE NPZ files")

    strategy = str(cfg["spatial"]["strategy"]).casefold()
    key = spatial_key(cfg)

    mode_dir = out / "fusion" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    oof = np.full(len(y), np.nan, dtype=np.float64)
    valid_fold_predictions = []
    fold_rows = []
    gate_rows = []
    scalar_audit_rows = []
    feature_root = spatial_feature_root(cfg, out)

    for fold in range(1, 6):
        fold_dir = mode_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        dev = np.flatnonzero(folds != fold)
        hold = np.flatnonzero(folds == fold)

        if set(groups[dev]) & set(groups[hold]):
            raise AssertionError("Patient leakage between development and holdout")

        temporal_train = temporal_valid = None
        scalar_audit = None
        if mode != "spatial_only":
            if use_scalar:
                scalar_train, scalar_valid, scalar_audit = clean_scalar_foldwise(
                    tr["scalar"], va["scalar"], dev, cfg
                )
                temporal_train = np.concatenate([tr["deep"], scalar_train], axis=1)
                temporal_valid = np.concatenate([va["deep"], scalar_valid], axis=1)
                scalar_audit_rows.append({"fold": fold, **scalar_audit})
            else:
                temporal_train = tr["deep"]
                temporal_valid = va["deep"]

        spatial_train = spatial_valid = None
        if mode != "cave_only":
            model_name = spatial_model_name(strategy, fold)
            spatial_train = load_spatial(
                feature_root,
                model_name,
                "Train",
                key,
                tr["series_uid"],
            )
            spatial_valid = load_spatial(
                feature_root,
                model_name,
                "Valid",
                key,
                va["series_uid"],
            )

        # No PCA / scaler here: raw upstream feature is learned-projected exactly
        # as specified by the teacher diagram.
        rel_tr, rel_va = patient_inner_split(
            y[dev],
            groups[dev],
            float(cfg["fusion"]["inner_val_fraction"]),
            int(cfg["fusion"]["seed"]) + fold,
        )
        inner_tr = dev[rel_tr]
        inner_va = dev[rel_va]

        best_epoch, best_inner_ap = select_epoch(
            mode,
            spatial_train,
            temporal_train,
            y,
            inner_tr,
            inner_va,
            cfg,
            device,
            int(cfg["fusion"]["seed"]) + fold * 100,
            fold_dir,
        )

        model = refit(
            mode,
            spatial_train,
            temporal_train,
            y,
            dev,
            best_epoch,
            cfg,
            device,
            int(cfg["fusion"]["seed"]) + fold * 10000,
            fold_dir,
        )

        hold_result = predict(
            model,
            spatial_train,
            temporal_train,
            hold,
            cfg,
            device,
            mode,
            collect_gates=True,
        )
        valid_result = predict(
            model,
            spatial_valid,
            temporal_valid,
            np.arange(len(yv)),
            cfg,
            device,
            mode,
            collect_gates=True,
        )

        oof[hold] = hold_result["prob"]
        valid_fold_predictions.append(valid_result["prob"])

        fold_rows.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "best_inner_AUPRC": best_inner_ap,
            "OOF_AUPRC": safe_ap(y[hold], hold_result["prob"]),
            "OOF_AUROC": safe_auc(y[hold], hold_result["prob"]),
            "OOF_Brier": brier(y[hold], hold_result["prob"]),
        })

        if "gate_2d" in hold_result:
            for j, global_idx in enumerate(hold):
                gate_rows.append({
                    "fold": fold,
                    "series_uid": str(tr["series_uid"][global_idx]),
                    "patient_id": str(tr["patient_id"][global_idx]),
                    "target": int(y[global_idx]),
                    "gate_2d_mean": float(hold_result["gate_2d"][j].mean()),
                    "gate_t_mean": float(hold_result["gate_t"][j].mean()),
                })

    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete OOF predictions")

    valid_prob = np.mean(np.stack(valid_fold_predictions), axis=0)

    pilot_warning = (
        strategy in {"pilot_single", "external_checkpoint"}
        and mode != "cave_only"
    )

    metrics = {
        "mode": mode,
        "strategy": strategy,
        "spatial_representation": cfg["spatial"].get("representation", "global_only"),
        "spatial_feature_root": str(feature_root),
        "temporal_representation": cfg.get("temporal", {}).get("representation", "deep_only"),
        "scalar_cleaning_by_outer_fold": scalar_audit_rows,
        "roi_source": cfg["spatial"].get("roi_source", ""),
        "train_oof": {
            "AUROC": safe_auc(y, oof),
            "AUPRC": safe_ap(y, oof),
            "Brier": brier(y, oof),
        },
        "valid": {
            "AUROC": safe_auc(yv, valid_prob),
            "AUPRC": safe_ap(yv, valid_prob),
            "Brier": brier(yv, valid_prob),
        },
        "teacher_projection_exact": True,
        "uses_pca": False,
        "pilot_representation_warning": pilot_warning,
        "representation_oof_status": "strict_crossfit" if strategy == "strict_crossfit" else "pilot_not_representation_crossfit",
        "segmentation_population": cfg["spatial"].get("strict_crossfit_segmentation_population") if strategy == "strict_crossfit" else cfg["spatial"].get("pilot_segmentation_population"),
        "valid_representation_status": "strict_train_only_representation" if strategy == "strict_crossfit" else "pilot_all_2d_includes_valid_image_mask",
        "valid_used_for_model_selection": False,
    }

    atomic_json(metrics, mode_dir / "metrics.json")
    atomic_csv(pd.DataFrame(fold_rows), mode_dir / "fold_metrics.csv")
    atomic_csv(
        pd.DataFrame({
            "series_uid": tr["series_uid"].astype(str),
            "patient_id": tr["patient_id"].astype(str),
            "target": y,
            "fold": folds,
            "probability": oof,
            "representation_oof_status": "strict_crossfit" if strategy == "strict_crossfit" else "pilot_not_representation_crossfit",
        }),
        mode_dir / "train_oof_predictions.csv",
    )
    atomic_csv(
        pd.DataFrame({
            "series_uid": va["series_uid"].astype(str),
            "patient_id": va["patient_id"].astype(str),
            "target": yv,
            "probability": valid_prob,
        }),
        mode_dir / "valid_predictions.csv",
    )
    if gate_rows:
        atomic_csv(pd.DataFrame(gate_rows), mode_dir / "oof_gate_summary.csv")
    if scalar_audit_rows:
        atomic_csv(pd.DataFrame(scalar_audit_rows), mode_dir / "scalar_cleaning_audit.csv")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_path(cfg["output_root"], cfg["project_root"])

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    modes = cfg["fusion"]["modes"] if args.mode == "all" else [args.mode]

    strategy = str(cfg["spatial"]["strategy"]).casefold()
    feature_root = spatial_feature_root(cfg, out)
    if strategy == "pilot_single":
        expected_feature_models = ["pilot"]
    elif strategy == "external_checkpoint":
        expected_feature_models = ["external"]
    else:
        expected_feature_models = [f"fold_{i}" for i in range(1,6)]

    for mode in modes:
        if mode != "cave_only":
            for name in expected_feature_models:
                if not (feature_root / "seg_features" / name / ".SUCCESS.json").is_file():
                    raise RuntimeError(
                        f"Missing spatial featurebank {name} under {feature_root}."
                    )
        run_mode(mode, cfg, out, device)


if __name__ == "__main__":
    main()
