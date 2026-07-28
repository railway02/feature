from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = get_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.loc[manifest["selected_for_extraction"].astype(bool)].copy()
    expected = 0
    success = 0
    failed = 0
    nonfinite = 0
    wrong_dim = 0
    rows = []
    for _, row in manifest.iterrows():
        for phase in ("pre", "post"):
            paths = str(row.get(f"{phase}_frame_paths", ""))
            if not paths or paths == "nan":
                continue
            expected += 1
            phase_dir = (
                args.feature_root / str(row["split"]).lower() /
                str(row["patient_id"]) / str(row["series_uid"]) / phase
            )
            if (phase_dir / ".SUCCESS").exists():
                success += 1
                vector = np.load(phase_dir / "embedding_5120.npy")
                wrong_dim += int(vector.shape != (5120,))
                nonfinite += int(not np.isfinite(vector).all())
            else:
                failed += 1
            rows.append({
                "series_uid": row["series_uid"],
                "phase": phase,
                "success": (phase_dir / ".SUCCESS").exists(),
            })
    report = {
        "expected_phase_count": expected,
        "success_phase_count": success,
        "failed_phase_count": failed,
        "wrong_dimension_count": wrong_dim,
        "nonfinite_embedding_count": nonfinite,
        "duplicate_series_uid": int(manifest["series_uid"].duplicated().sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(report, indent=2))
    if failed or wrong_dim or nonfinite or report["duplicate_series_uid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
