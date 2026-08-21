#!/usr/bin/env python
"""Build per-series technical QC contact sheets from production PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PANELS = [
    "intra_pre_peak_selection.png", "intra_post_peak_selection.png",
    "intra_pre_motion_overlay.png", "intra_post_motion_overlay.png",
    "pre_reference_to_peak_mask_overlay.png", "post_reference_to_peak_mask_overlay.png",
    "global_before.png", "global_similarity.png", "global_primary_local_roi.png",
    "nonrigid_overlay.png", "displacement.png", "canonical_logjac.png",
]


def panel(path: Path, size: int) -> np.ndarray:
    x = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if x is None:
        x = np.full((size, size, 3), 235, np.uint8)
        cv2.putText(x, "MISSING", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 0, 180), 2, cv2.LINE_AA)
    else:
        scale = min(size / x.shape[0], size / x.shape[1])
        resized = cv2.resize(x, (max(1, int(x.shape[1] * scale)),
                                 max(1, int(x.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size, 3), 255, np.uint8)
        y = (size - resized.shape[0]) // 2; z = (size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], z:z + resized.shape[1]] = resized
        x = canvas
    cv2.rectangle(x, (0, size - 30), (size, size), (255, 255, 255), -1)
    cv2.putText(x, path.stem[:38], (5, size - 9), cv2.FONT_HERSHEY_SIMPLEX,
                .42, (20, 20, 20), 1, cv2.LINE_AA)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--panel-size", type=int, default=300)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    count = 0
    for features in sorted(Path(a.output_root).glob("*/*/*/features.json")):
        case = features.parent; qc = case / "qc"
        tiles = [panel(qc / name, a.panel_size) for name in PANELS]
        rows = [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, 12, 3)]
        sheet = np.concatenate(rows, axis=0)
        target = out / f"{case.parent.name}__{case.name}.png"
        cv2.imwrite(str(target), sheet)
        count += 1
    print(f"wrote {count} technical/model QC contact sheets; no clinician validation claimed")


if __name__ == "__main__":
    main()
