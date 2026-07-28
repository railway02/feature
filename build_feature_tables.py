from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_phase(root: Path, split: str, patient_id: str, series_uid: str, phase: str):
    phase_dir = root / split.lower() / patient_id / series_uid / phase
    if not (phase_dir / ".SUCCESS").exists():
        return None, {}, None
    embedding = np.load(phase_dir / "embedding_5120.npy").astype(np.float32)
    scalar = json.loads((phase_dir / "scalar_features.json").read_text(encoding="utf-8"))
    metadata = json.loads((phase_dir / "metadata.json").read_text(encoding="utf-8"))
    return embedding, scalar, metadata


def main() -> int:
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.loc[manifest["selected_for_extraction"].astype(bool)].copy()

    embeddings = []
    scalar_rows = []
    keys = []
    for _, row in manifest.iterrows():
        pid = str(row["patient_id"])
        uid = str(row["series_uid"])
        split = str(row["split"])
        pre, pre_scalar, _ = load_phase(args.feature_root, split, pid, uid, "pre")
        post, post_scalar, _ = load_phase(args.feature_root, split, pid, uid, "post")
        missing_pre = pre is None
        missing_post = post is None
        pre = np.full(5120, np.nan, np.float32) if pre is None else pre
        post = np.full(5120, np.nan, np.float32) if post is None else post
        embeddings.append(np.stack([pre, post], axis=0))
        keys.append(uid)
        scalar_row = {
            "patient_id": pid,
            "series_uid": uid,
            "split": split,
            "missing_pre": int(missing_pre),
            "missing_post": int(missing_post),
        }
        scalar_row.update({f"pre_{k}": v for k, v in pre_scalar.items()})
        scalar_row.update({f"post_{k}": v for k, v in post_scalar.items()})
        scalar_rows.append(scalar_row)

    array = np.stack(embeddings, axis=0)
    np.savez_compressed(
        args.output_dir / "series_embeddings_5120.npz",
        series_uid=np.asarray(keys),
        embeddings=array,
    )
    scalar_df = pd.DataFrame(scalar_rows)
    scalar_df.to_csv(args.output_dir / "series_scalar_features.csv", index=False)
    try:
        scalar_df.to_parquet(args.output_dir / "series_scalar_features.parquet", index=False)
    except Exception as exc:
        print(f"[WARN] parquet not written: {exc}")
    print(f"[PASS] rows={len(scalar_df)} embeddings={array.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
