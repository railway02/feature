from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PHASE_FILES = (
    "embedding_5120.npy", "embedding_views_5120.npz",
    "probabilities_original.fp16.npz", "curves.npz",
    "scalar_features.json", "metadata.json", "qc.json", ".SUCCESS.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-series", type=int, required=True)
    parser.add_argument("--expected-pre", type=int, required=True)
    parser.add_argument("--expected-post", type=int, required=True)
    parser.add_argument("--expected-config-hash", required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.loc[manifest["selected_for_extraction"].astype(bool)].copy()
    report = {
        "series": len(manifest), "expected_series": args.expected_series,
        "expected_pre": args.expected_pre, "expected_post": args.expected_post,
        "success_pre": 0, "success_post": 0, "missing_files": 0,
        "wrong_embedding_shape": 0, "nonfinite_embeddings": 0,
        "wrong_scalar_count": 0, "wrong_config_hash": 0,
        "duplicate_series_uid": int(manifest["series_uid"].duplicated().sum()),
        "all_zero_embeddings": 0,
    }
    vectors = []
    details = []
    for _, row in manifest.iterrows():
        for phase in ("pre", "post"):
            paths = str(row.get(f"{phase}_frame_paths", ""))
            if not paths or paths.lower() == "nan":
                continue
            directory = args.feature_root / str(row["split"]).lower() / str(row["patient_id"]) / str(row["series_uid"]) / phase
            missing = [name for name in REQUIRED_PHASE_FILES if not (directory / name).exists()]
            if missing:
                report["missing_files"] += len(missing)
                details.append({"series_uid": row["series_uid"], "phase": phase, "missing": "|".join(missing)})
                continue
            success = json.loads((directory / ".SUCCESS.json").read_text(encoding="utf-8"))
            report["wrong_config_hash"] += int(success.get("frozen_config_hash") != args.expected_config_hash)
            vector = np.load(directory / "embedding_5120.npy")
            report["wrong_embedding_shape"] += int(vector.shape != (5120,))
            report["nonfinite_embeddings"] += int(not np.isfinite(vector).all())
            report["all_zero_embeddings"] += int(np.allclose(vector, 0))
            vectors.append(vector)
            scalars = json.loads((directory / "scalar_features.json").read_text(encoding="utf-8"))
            report["wrong_scalar_count"] += int(len(scalars) != 206)
            report[f"success_{phase}"] += 1
    report["embedding_median_channel_variance"] = float(np.median(np.var(np.stack(vectors), axis=0))) if vectors else 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(details).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(report, indent=2))
    hard_fail = (
        report["series"] != args.expected_series or
        report["success_pre"] != args.expected_pre or
        report["success_post"] != args.expected_post or
        any(report[key] for key in (
            "missing_files", "wrong_embedding_shape", "nonfinite_embeddings",
            "wrong_scalar_count", "wrong_config_hash", "duplicate_series_uid", "all_zero_embeddings",
        ))
    )
    return 2 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
