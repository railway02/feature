#!/usr/bin/env python3
"""Masked spatial-only DeepLab-ASPP versus SegResNet outcome experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset


FAMILIES = ("deeplabv3plus_resnet50_imagenet", "segresnet")


def setup(cfg_path: str):
    raw = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    sys.path.insert(0, raw["sources"]["v6_code_root"])
    from common import atomic_csv, atomic_json, atomic_torch_save, canonical_hash, load_config, sha256_file, set_seed
    from model_interface import build_model
    cfg = load_config(cfg_path)
    return cfg, atomic_csv, atomic_json, atomic_torch_save, canonical_hash, sha256_file, set_seed, build_model


def normalize_image(image: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Image contains no finite pixels")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def prepare_image_with_valid_mask(path: str, cfg: dict):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = normalize_image(image, cfg["data"]["percentile_low"], cfg["data"]["percentile_high"])
    height, width = image.shape
    target = int(cfg["data"]["input_size"])
    scale = min(target / height, target / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(np.median(resized)))
    valid = np.zeros((target, target), dtype=np.float32)
    valid[top:top + new_h, left:left + new_w] = 1.0
    audit = (scale, top, bottom, left, right, float(valid.mean()))
    return padded.astype(np.float32), valid, audit


class PhaseImageDataset(Dataset):
    def __init__(self, cases: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.rows = []
        for row in cases.itertuples(index=False):
            for phase in cfg["data"]["phases"]:
                self.rows.append((str(row.series_uid), phase, str(getattr(row, f"{phase.lower()}_image"))))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        _, _, path = self.rows[index]
        image, valid, audit = prepare_image_with_valid_mask(path, self.cfg)
        return torch.from_numpy(image[None]), torch.from_numpy(valid[None]), int(index), torch.tensor(audit, dtype=torch.float32)


def deeplab_raw_aspp_and_logits(model, x: torch.Tensor):
    """Replicate SMP decoder exactly while exposing the raw ASPP tensor."""
    decoder = model.model.decoder
    features = model.model.encoder(model.normalize_input(x))
    aspp_raw = decoder.aspp(features[-1])
    aspp_up = decoder.up(aspp_raw)
    high_res = decoder.block1(features[-4])
    fused = decoder.block2(torch.cat([aspp_up, high_res], dim=1))
    logits = model.model.segmentation_head(fused)
    return aspp_raw, logits


def model_features_and_logits(model, family: str, x: torch.Tensor):
    if family == "deeplabv3plus_resnet50_imagenet":
        return deeplab_raw_aspp_and_logits(model, x)
    return model.encode_and_decode(x)


def masked_pool(feature: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor):
    """Float32 masked global and soft predicted-ROI pooling."""
    feature = feature.float()
    probability = torch.sigmoid(logits.float())
    valid_f = F.interpolate(valid.float(), size=feature.shape[-2:], mode="area")
    probability_f = F.interpolate(probability, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    valid_mass = valid_f.sum(dim=(-2, -1))
    roi_weight = probability_f * valid_f
    roi_mass = roi_weight.sum(dim=(-2, -1))
    global_feature = (feature * valid_f).sum(dim=(-2, -1)) / valid_mass.clamp_min(1e-6)
    roi_feature = (feature * roi_weight).sum(dim=(-2, -1)) / roi_mass.clamp_min(1e-6)
    return global_feature, roi_feature, valid_mass, roi_mass


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def case_manifest(cfg: dict) -> pd.DataFrame:
    cases = pd.read_csv(cfg["sources"]["case_manifest"], dtype={"patient_id": str, "series_uid": str})
    train = cases.loc[cases["split"].eq("Train")].copy().reset_index(drop=True)
    train["fold"] = pd.to_numeric(train["fold"], errors="raise").astype(int)
    train["target"] = pd.to_numeric(train["target"], errors="raise").astype(int)
    if len(train) != int(cfg["data"]["expected_train_series"]) or train.series_uid.nunique() != len(train):
        raise RuntimeError("Unexpected outcome Train cases")
    if train.groupby("patient_id")["fold"].nunique().max() != 1:
        raise RuntimeError("A patient belongs to multiple outcome folds")
    counts = train.fold.value_counts().sort_index().to_dict()
    expected = {int(k): int(v) for k, v in cfg["data"]["expected_fold_counts"].items()}
    if counts != expected:
        raise RuntimeError(f"Unexpected outcome fold counts: {counts}")
    return train


def checkpoint_path(cfg: dict, family: str, fold: int) -> Path:
    return Path(cfg["sources"]["v6_segmentation_root"]) / family / f"fold_{fold}" / "model.pt"


def preflight(cfg: dict, atomic_json, sha256_file, build_model, device: torch.device):
    if any("temporal" in key.lower() or "cave" in key.lower() for key in cfg["sources"]):
        raise RuntimeError("This spatial-only config must not declare CAVE/temporal inputs")
    train = case_manifest(cfg)
    checkpoints = {}
    for family in FAMILIES:
        for fold in range(1, 6):
            path = checkpoint_path(cfg, family, fold)
            success = path.with_name("SUCCESS.json")
            if not path.is_file() or not success.is_file():
                raise FileNotFoundError(f"Missing frozen segmentation asset: {path}")
            checkpoints[f"{family}/fold_{fold}"] = {"path": str(path), "sha256": sha256_file(path)}
    model = build_model("deeplabv3plus_resnet50_imagenet", cfg, load_pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint_path(cfg, "deeplabv3plus_resnet50_imagenet", 1), map_location="cpu")["state_dict"], strict=True)
    x = torch.zeros(1, 1, int(cfg["data"]["input_size"]), int(cfg["data"]["input_size"]), device=device)
    with torch.no_grad():
        raw, manual_logits = deeplab_raw_aspp_and_logits(model, x)
        _, native_logits = model.encode_and_decode(x)
    max_abs = float((manual_logits - native_logits).abs().max().cpu())
    expected = tuple(cfg["feature_protocol"]["deeplab_expected_feature_shape"])
    if tuple(raw.shape[1:]) != expected or max_abs > 1e-6:
        raise RuntimeError(f"DeepLab ASPP/logit contract failed: raw={tuple(raw.shape)}, max_abs={max_abs}")
    report = {
        "status": "success", "experiment": "v7_masked_spatial_only", "train_rows": len(train),
        "cave_used": False, "gt_masks_used": False, "checkpoint_hashes": checkpoints,
        "deeplab_raw_aspp_shape": list(raw.shape[1:]), "deeplab_manual_vs_native_logits_max_abs": max_abs,
        "feature_protocol": cfg["feature_protocol"], "config_sha256": cfg["_config_sha256"],
    }
    atomic_json(report, Path(cfg["report_root"]) / "preflight" / "SUCCESS.json")


def extract_train(cfg: dict, atomic_json, build_model, device: torch.device):
    cases = case_manifest(cfg)
    dataset = PhaseImageDataset(cases, cfg)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(cfg["data"]["num_workers"]) > 0)
    lookup = {(uid, phase): index for index, (uid, phase, _) in enumerate(dataset.rows)}
    for family in FAMILIES:
        for fold in range(1, 6):
            model = build_model(family, cfg, load_pretrained=False).to(device).eval()
            model.load_state_dict(torch.load(checkpoint_path(cfg, family, fold), map_location="cpu")["state_dict"], strict=True)
            g = r = valid_mass = roi_mass = None
            for x, valid, index, _ in loader:
                x, valid = x.to(device, non_blocking=True), valid.to(device, non_blocking=True)
                with torch.no_grad(), autocast(enabled=device.type == "cuda"):
                    fmap, logits = model_features_and_logits(model, family, x)
                global_feature, roi_feature, vmass, rmass = masked_pool(fmap, logits, valid)
                if float(rmass.min().cpu()) <= float(cfg["feature_protocol"]["min_predroi_mass"]):
                    raise RuntimeError(f"{family} fold {fold}: near-zero predicted ROI mass")
                if g is None:
                    n = len(dataset)
                    g = np.empty((n, global_feature.shape[1]), dtype=np.float32)
                    r = np.empty((n, roi_feature.shape[1]), dtype=np.float32)
                    valid_mass = np.empty(n, dtype=np.float32)
                    roi_mass = np.empty(n, dtype=np.float32)
                ii = index.numpy()
                g[ii] = global_feature.cpu().numpy(); r[ii] = roi_feature.cpu().numpy()
                valid_mass[ii] = vmass.squeeze(1).cpu().numpy(); roi_mass[ii] = rmass.squeeze(1).cpu().numpy()
            rows = []
            for row in cases.itertuples(index=False):
                pre, post = lookup[(str(row.series_uid), "Pre")], lookup[(str(row.series_uid), "Post")]
                rows.append(np.concatenate([g[pre], r[pre], g[post], r[post]]))
            spatial = np.stack(rows).astype(np.float32)
            if spatial.shape != (len(cases), int(cfg["feature_protocol"]["case_feature_dimension"])) or not np.isfinite(spatial).all():
                raise RuntimeError(f"Invalid feature matrix for {family} fold {fold}: {spatial.shape}")
            out = Path(cfg["output_root"]) / "features" / family / f"fold_{fold}"
            atomic_npz(out / "train.npz", spatial_feature=spatial, series_uid=cases.series_uid.to_numpy(dtype=str), patient_id=cases.patient_id.to_numpy(dtype=str), target=cases.target.to_numpy(dtype=np.int64), outer_fold=cases.fold.to_numpy(dtype=np.int64), phase_valid_mass=valid_mass.reshape(len(cases), 2), phase_predroi_mass=roi_mass.reshape(len(cases), 2))
            atomic_json({"status": "success", "family": family, "fold": fold, "rows": len(cases), "feature_shape": list(spatial.shape), "feature_tap": cfg["feature_protocol"]["deeplab_feature_tap"] if family.startswith("deeplab") else cfg["feature_protocol"]["segresnet_feature_tap"], "pooling": cfg["feature_protocol"]["pooling"], "gt_masks_used": False, "cave_used": False, "min_predroi_mass": float(roi_mass.min()), "max_predroi_mass": float(roi_mass.max())}, out / "SUCCESS.json")
            del model


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def verify(cfg: dict, atomic_json, sha256_file, device: torch.device):
    del device
    for family in FAMILIES:
        banks = [load_npz(Path(cfg["output_root"]) / "features" / family / f"fold_{fold}" / "train.npz") for fold in range(1, 6)]
        reference = banks[0]
        checks = {
            "five_folds_present": len(banks) == 5,
            "rows": all(x["spatial_feature"].shape == (781, 1024) for x in banks),
            "unicode_ids": all(x["series_uid"].dtype.kind in "US" and x["patient_id"].dtype.kind in "US" for x in banks),
            "same_uid_order": all(np.array_equal(reference["series_uid"], x["series_uid"]) for x in banks[1:]),
            "same_target_and_fold": all(np.array_equal(reference["target"], x["target"]) and np.array_equal(reference["outer_fold"], x["outer_fold"]) for x in banks[1:]),
            "unique_uids": len(np.unique(reference["series_uid"])) == 781,
            "finite_features": all(np.isfinite(x["spatial_feature"]).all() for x in banks),
            "valid_mass_positive": all((x["phase_valid_mass"] > 0).all() for x in banks),
            "predroi_mass_above_minimum": all((x["phase_predroi_mass"] > float(cfg["feature_protocol"]["min_predroi_mass"])).all() for x in banks),
            "no_gt_fields": all(not any(key.startswith("gt") for key in x) for x in banks),
            "no_temporal_fields": all(not any("temporal" in key or "cave" in key for key in x) for x in banks),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Feature verification failed for {family}: {checks}")
        atomic_json({"status": "PASS", "family": family, "checks": checks, "feature_files_sha256": {str(fold): sha256_file(Path(cfg["output_root"]) / "features" / family / f"fold_{fold}" / "train.npz") for fold in range(1, 6)}}, Path(cfg["output_root"]) / "featurebanks" / family / "verification.json")


def metrics(y, probability):
    return {"auroc": float(roc_auc_score(y, probability)), "auprc": float(average_precision_score(y, probability)), "brier": float(brier_score_loss(y, probability))}


def patient_split(y, patient_id, fraction, seed):
    frame = pd.DataFrame({"patient_id": np.asarray(patient_id).astype(str), "target": np.asarray(y).astype(int)}).groupby("patient_id", as_index=False).agg(target=("target", "max"))
    train, valid = next(StratifiedShuffleSplit(n_splits=1, test_size=float(fraction), random_state=int(seed)).split(frame.patient_id, frame.target))
    train_ids, valid_ids = set(frame.iloc[train].patient_id), set(frame.iloc[valid].patient_id)
    ids = np.asarray(patient_id).astype(str)
    return np.flatnonzero(np.isin(ids, list(train_ids))), np.flatnonzero(np.isin(ids, list(valid_ids)))


def fit_scaler(x, index, epsilon):
    mean = x[index].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[index].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < epsilon] = 1.0
    return mean, std


def transform(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def outcome_loader(x, y, index, batch, shuffle, seed):
    data = TensorDataset(torch.from_numpy(x[index]).float(), torch.from_numpy(y[index].astype(np.float32)).view(-1, 1))
    return DataLoader(data, batch_size=int(batch), shuffle=bool(shuffle), generator=torch.Generator().manual_seed(int(seed)), drop_last=False)


def train_epoch(model, loader, optimizer, scaler, device, pos_weight, amp):
    model.train(); losses = []; weight = torch.tensor([pos_weight], device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            loss = F.binary_cross_entropy_with_logits(model(spatial=x)["logit"], y, pos_weight=weight)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, x, y, index, batch, device):
    model.eval(); values = []
    for xx, _ in outcome_loader(x, y, index, batch, False, 0):
        values.append(torch.sigmoid(model(spatial=xx.to(device))["logit"]).cpu().numpy().ravel())
    return np.concatenate(values)


def outcome_oof(cfg, atomic_csv, atomic_json, atomic_torch_save, canonical_hash, set_seed, device: torch.device):
    sys.path.insert(0, cfg["sources"]["v5_code_root"])
    from fusion_models import OutcomeModel
    fusion = cfg["outcome"]
    for family in FAMILIES:
        banks = [load_npz(Path(cfg["output_root"]) / "features" / family / f"fold_{fold}" / "train.npz") for fold in range(1, 6)]
        base = banks[0]; y, folds, patient = base["target"].astype(int), base["outer_fold"].astype(int), base["patient_id"].astype(str)
        oof = np.full(len(y), np.nan, dtype=np.float32); rows = []
        for fold in range(1, 6):
            x = banks[fold - 1]["spatial_feature"].astype(np.float32)
            development, holdout = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            if set(patient[development]) & set(patient[holdout]):
                raise RuntimeError(f"{family} fold {fold}: patient leakage")
            tr_rel, va_rel = patient_split(y[development], patient[development], fusion["inner_val_fraction"], int(fusion["seed"]) + fold)
            inner_train, inner_valid = development[tr_rel], development[va_rel]
            search_mean, search_std = fit_scaler(x, inner_train, fusion["scaler_epsilon"]); x_search = transform(x, search_mean, search_std)
            fold_dir = Path(cfg["output_root"]) / "outcome_oof" / family / f"fold_{fold}"; fold_dir.mkdir(parents=True, exist_ok=True)
            set_seed(int(fusion["seed"]) + fold * 100)
            # The frozen v5 constructor requires temporal_dim syntactically,
            # but spatial_only never instantiates or receives a temporal branch.
            model = OutcomeModel(mode="spatial_only", spatial_dim=1024, temporal_dim=1, hidden_dim=int(fusion["hidden_dim"]), dropout=float(fusion["dropout"])).to(device)
            optimizer = AdamW(model.parameters(), lr=float(fusion["learning_rate"]), weight_decay=float(fusion["weight_decay"])); amp = bool(fusion["amp"] and device.type == "cuda"); scaler = GradScaler(enabled=amp)
            positive = max(1, int(y[inner_train].sum())); pos_weight = max(1, len(inner_train) - positive) / positive
            best_epoch, best_ap, bad, history = 0, -1.0, 0, []
            train_loader = outcome_loader(x_search, y, inner_train, fusion["batch_size"], True, int(fusion["seed"]) + fold * 100)
            for epoch in range(1, int(fusion["max_epochs"]) + 1):
                loss = train_epoch(model, train_loader, optimizer, scaler, device, pos_weight, amp)
                score = metrics(y[inner_valid], predict(model, x_search, y, inner_valid, fusion["batch_size"], device))
                history.append({"epoch": epoch, "train_loss": loss, "inner_valid_AUPRC": score["auprc"], "inner_valid_AUROC": score["auroc"]})
                if score["auprc"] > best_ap + 1e-6: best_epoch, best_ap, bad = epoch, score["auprc"], 0
                else: bad += 1
                if epoch >= int(fusion["min_epochs"]) and bad >= int(fusion["patience"]): break
            atomic_csv(pd.DataFrame(history), fold_dir / "epoch_search.csv"); atomic_npz(fold_dir / "search_scaler.npz", mean=search_mean, std=search_std)
            del model
            refit_mean, refit_std = fit_scaler(x, development, fusion["scaler_epsilon"]); x_refit = transform(x, refit_mean, refit_std)
            set_seed(int(fusion["seed"]) + fold * 10000)
            model = OutcomeModel(mode="spatial_only", spatial_dim=1024, temporal_dim=1, hidden_dim=int(fusion["hidden_dim"]), dropout=float(fusion["dropout"])).to(device)
            optimizer = AdamW(model.parameters(), lr=float(fusion["learning_rate"]), weight_decay=float(fusion["weight_decay"])); scaler = GradScaler(enabled=amp)
            positive = max(1, int(y[development].sum())); pos_weight = max(1, len(development) - positive) / positive
            history = []; train_loader = outcome_loader(x_refit, y, development, fusion["batch_size"], True, int(fusion["seed"]) + fold * 10000)
            for epoch in range(1, best_epoch + 1):
                history.append({"epoch": epoch, "train_loss": train_epoch(model, train_loader, optimizer, scaler, device, pos_weight, amp)})
            probability = predict(model, x_refit, y, holdout, fusion["batch_size"], device); oof[holdout] = probability
            atomic_csv(pd.DataFrame(history), fold_dir / "fresh_refit_history.csv"); atomic_npz(fold_dir / "refit_scaler.npz", mean=refit_mean, std=refit_std)
            atomic_torch_save({"state_dict": model.state_dict(), "outer_fold": fold, "selected_epoch": best_epoch, "mode": "spatial_only", "spatial_dim": 1024, "temporal_input_used": False, "outcome_config": fusion, "fresh_refit_on_outer_development": True}, fold_dir / "model.pt")
            fold_metric = metrics(y[holdout], probability); rows.append({"fold": fold, "selected_epoch": best_epoch, "best_inner_AUPRC": best_ap, **{f"OOF_{key}": value for key, value in fold_metric.items()}})
            del model
        if not np.isfinite(oof).all(): raise RuntimeError(f"{family}: incomplete OOF")
        report = Path(cfg["report_root"]) / "outcome_oof" / family
        atomic_csv(pd.DataFrame({"series_uid": base["series_uid"], "patient_id": patient, "target": y, "outer_fold": folds, "probability": oof}), report / "train_oof_predictions.csv")
        atomic_csv(pd.DataFrame(rows), report / "fold_metrics.csv")
        atomic_json({"status": "success", "family": family, "mode": "spatial_only", "temporal_input_used": False, "cave_used": False, "gt_masks_used": False, "feature_protocol": cfg["feature_protocol"], "outcome_config_sha256": canonical_hash(fusion), "standardization": "inner-train-only for epoch search; outer-development-only for fresh refit", "train_oof": metrics(y, oof)}, report / "SUCCESS.json")


def compare(cfg, atomic_json, atomic_csv, device: torch.device):
    del device
    reports = {family: pd.read_csv(Path(cfg["report_root"]) / "outcome_oof" / family / "train_oof_predictions.csv", dtype={"patient_id": str, "series_uid": str}) for family in FAMILIES}
    deep, seg = reports["deeplabv3plus_resnet50_imagenet"], reports["segresnet"]
    for column in ("series_uid", "patient_id", "target", "outer_fold"):
        if not np.array_equal(deep[column].to_numpy(), seg[column].to_numpy()): raise RuntimeError(f"Comparison UID/target mismatch: {column}")
    y, pd_, ps = deep.target.to_numpy(), deep.probability.to_numpy(), seg.probability.to_numpy()
    rng = np.random.default_rng(int(cfg["bootstrap"]["seed"])); draws = int(cfg["bootstrap"]["draws"]); n = len(y); deltas = {key: [] for key in ("auroc", "auprc", "brier")}
    for _ in range(draws):
        index = rng.integers(0, n, n)
        if len(np.unique(y[index])) < 2: continue
        a, b = metrics(y[index], pd_[index]), metrics(y[index], ps[index])
        for key in deltas: deltas[key].append(a[key] - b[key])
    summary = {"status": "success", "comparison": "deeplab_raw_aspp_masked_minus_segresnet_bottleneck_masked", "deeplab": metrics(y, pd_), "segresnet": metrics(y, ps), "paired_bootstrap_delta": {key: {"point": metrics(y, pd_)[key] - metrics(y, ps)[key], "ci95": [float(np.quantile(value, .025)), float(np.quantile(value, .975))]} for key, value in deltas.items()}, "draws": draws, "cave_used": False}
    root = Path(cfg["report_root"]) / "comparison"; atomic_json(summary, root / "SPATIAL_ONLY_OOF_COMPARISON.json")
    atomic_csv(pd.DataFrame({"series_uid": deep.series_uid, "patient_id": deep.patient_id, "target": y, "outer_fold": deep.outer_fold, "deeplab_probability": pd_, "segresnet_probability": ps}), root / "paired_oof_predictions.csv")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("preflight", "extract-train", "verify", "outcome-oof", "compare")); parser.add_argument("--config", required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    cfg, atomic_csv, atomic_json, atomic_torch_save, canonical_hash, sha256_file, set_seed, build_model = setup(args.config)
    device = torch.device(args.device); os.environ["TORCH_HOME"] = cfg["torch_home"]
    if args.stage == "preflight": preflight(cfg, atomic_json, sha256_file, build_model, device)
    elif args.stage == "extract-train": extract_train(cfg, atomic_json, build_model, device)
    elif args.stage == "verify": verify(cfg, atomic_json, sha256_file, device)
    elif args.stage == "outcome-oof": outcome_oof(cfg, atomic_csv, atomic_json, atomic_torch_save, canonical_hash, set_seed, device)
    else: compare(cfg, atomic_json, atomic_csv, device)


if __name__ == "__main__": main()
