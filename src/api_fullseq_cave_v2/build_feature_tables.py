from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BLOCKS = (
    "f5_global_mean", "f5_vessel_mean", "f5_artery_mean", "f5_vein_mean",
    "f5_active_vessel_mean", "f5_vessel_top10_abs_magnitude",
    "f4_vessel_mean", "f4_artery_mean", "f4_active_vessel_mean",
    "f4_vessel_top10_abs_magnitude",
)


def load_phase(root: Path, row: pd.Series, phase: str):
    directory = root / str(row["split"]).lower() / str(row["patient_id"]) / str(row["series_uid"]) / phase
    if not (directory / ".SUCCESS.json").exists():
        return None, {}, None
    return (
        np.load(directory / "embedding_5120.npy").astype(np.float32),
        json.loads((directory / "scalar_features.json").read_text(encoding="utf-8")),
        json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
    )


def group_distances(pre: np.ndarray, post: np.ndarray) -> dict[str, float]:
    output = {}
    for index, name in enumerate(BLOCKS):
        a, b = pre[index * 512:(index + 1) * 512], post[index * 512:(index + 1) * 512]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        output[f"{name}_prepost_cosine"] = float(np.dot(a, b) / max(na * nb, 1e-8))
        output[f"{name}_prepost_normalized_l2"] = float(np.linalg.norm(a / max(na, 1e-8) - b / max(nb, 1e-8)))
        output[f"{name}_prepost_log_norm_ratio"] = float(np.log(max(nb, 1e-8) / max(na, 1e-8)))
        output[f"{name}_prepost_mean_abs_difference"] = float(np.mean(np.abs(a - b)))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.loc[manifest["selected_for_extraction"].astype(bool)].copy()
    embeddings, scalar_rows, keys = [], [], []
    for _, row in manifest.iterrows():
        pre, pre_scalar, _ = load_phase(args.feature_root, row, "pre")
        post, post_scalar, _ = load_phase(args.feature_root, row, "post")
        missing_pre, missing_post = pre is None, post is None
        pre_array = np.full(5120, np.nan, np.float32) if missing_pre else pre
        post_array = np.full(5120, np.nan, np.float32) if missing_post else post
        embeddings.append(np.stack([pre_array, post_array]))
        keys.append(str(row["series_uid"]))
        scalar = {
            "patient_id": str(row["patient_id"]), "series_uid": str(row["series_uid"]),
            "split": str(row["split"]), "source_type": str(row.get("source_type", "")),
            "series_id": str(row.get("series_id", "")),
            "missing_pre": int(missing_pre), "missing_post": int(missing_post),
        }
        scalar.update({f"pre_{k}": v for k, v in pre_scalar.items()})
        scalar.update({f"post_{k}": v for k, v in post_scalar.items()})
        if not missing_pre and not missing_post:
            scalar.update(group_distances(pre, post))
        scalar_rows.append(scalar)
    array = np.stack(embeddings)
    np.savez_compressed(args.output_dir / "series_embeddings_5120.npz", series_uid=np.asarray(keys), embeddings=array)
    frame = pd.DataFrame(scalar_rows)
    frame.to_csv(args.output_dir / "series_scalar_features.csv", index=False)
    frame.to_parquet(args.output_dir / "series_scalar_features.parquet", index=False)
    print(f"[PASS] series={len(frame)} embeddings={array.shape} scalar_columns={len(frame.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
