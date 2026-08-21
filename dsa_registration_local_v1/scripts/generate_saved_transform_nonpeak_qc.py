#!/usr/bin/env python3
"""Read-only QC of saved temporal frame-to-peak transforms.

This script deliberately does not estimate a transform.  It reconstructs native
local frames and applies the already saved .tfm files to make non-peak-frame QC.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json, load_config
from dsa_local_reg.local_geometry import BBox, crop_with_border_median_padding
from dsa_local_reg.v5_adapter import load_v5_module


def read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None: raise FileNotFoundError(path)
    return image


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = a[mask].astype(float), b[mask].astype(float)
    if len(x) < 16 or np.std(x) < 1e-8 or np.std(y) < 1e-8: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def normal(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    lo, hi = np.percentile(values, [1, 99])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(a: np.ndarray, peak: np.ndarray) -> np.ndarray:
    # red=current, green=frozen peak; yellow means agreement.
    return np.dstack([normal(a), normal(peak), np.zeros_like(a, dtype=float)])


def phase_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    frames = d['frames']
    rot = np.array([f['canonical_parameters']['rotation_deg'] for f in frames], float)
    tx = np.array([f['canonical_parameters']['tx'] for f in frames], float)
    ty = np.array([f['canonical_parameters']['ty'] for f in frames], float)
    positions = np.array([f['manifest_frame_position'] for f in frames], int)
    peak = int(d['frozen_peak_index'])
    nonpeak = positions != peak
    return {'path': path, 'data': d, 'max_abs_rotation_deg': float(np.max(np.abs(rot[nonpeak]))),
            'max_displacement_px': float(np.max(np.hypot(tx[nonpeak], ty[nonpeak])))}


def choose(root: Path) -> list[dict]:
    records = [phase_summary(path) for path in root.glob('cases/*/*/temporal/*/intra_registration.json')]
    df = pd.DataFrame([{'key': str(r['path']), 'series_uid': r['path'].parts[-4], 'split': r['path'].parts[-5],
                        'phase': r['path'].parts[-2], 'max_abs_rotation_deg': r['max_abs_rotation_deg'],
                        'max_displacement_px': r['max_displacement_px']} for r in records])
    by_series = df.sort_values(['max_abs_rotation_deg', 'phase'], ascending=[False, True]).drop_duplicates('series_uid')
    top = by_series.head(10).assign(group='largest_rotation')
    median = float(by_series.max_abs_rotation_deg.median())
    normal_rows = by_series[~by_series.series_uid.isin(top.series_uid)].assign(distance=lambda x: (x.max_abs_rotation_deg-median).abs()).sort_values(['distance','series_uid']).head(5).assign(group='ordinary_near_median').drop(columns='distance')
    lookup = {str(r['path']): r for r in records}
    selected = pd.concat([top, normal_rows], ignore_index=True)
    return [{**row, 'record': lookup[row['key']]} for row in selected.to_dict('records')]


def selected_indices(frames: list[dict], peak: int) -> list[int]:
    positions = np.array([f['manifest_frame_position'] for f in frames], int)
    nonpeak = positions[positions != peak]
    early = int(nonpeak[0]); late = int(nonpeak[-1])
    near = int(nonpeak[np.argmin(np.abs(nonpeak - peak))])
    return [early, near, late]


def make_case(item: dict, output: Path, sitkmod) -> dict:
    source = item['record']['data']; frames = source['frames']; contract = json.loads((item['record']['path'].parent/'local_sequence_contract.json').read_text())
    bbox = BBox.from_text(contract['expanded_bbox']); paths = contract['frame_paths']
    raw_by_position = {i: crop_with_border_median_padding(read_gray(path), bbox) for i, path in enumerate(paths)}
    peak_pos = int(source['frozen_peak_index']); peak = raw_by_position[peak_pos].image.astype(np.float32); crop_valid = raw_by_position[peak_pos].valid_support
    transformed, supports, per_frame = {}, [], []
    for frame in frames:
        pos = int(frame['manifest_frame_position']); raw = raw_by_position[pos].image.astype(np.float32)
        transform = __import__('SimpleITK').ReadTransform(str(item['record']['path'].parent/'transforms'/frame['transform_path']))
        corrected = sitkmod.resample(raw, peak, transform, default=float(np.median(np.r_[raw[0],raw[-1],raw[:,0],raw[:,-1]])))
        valid = sitkmod.resample(crop_valid.astype(np.uint8), peak, transform, is_mask=True, default=0.0)
        transformed[pos] = corrected
        supports.append(valid)
        per_frame.append({**frame, 'raw_ncc_to_peak': ncc(raw, peak, crop_valid), 'corrected_ncc_to_peak': ncc(corrected, peak, valid & crop_valid)})
    stable = np.logical_and.reduce(supports)
    picks = selected_indices(frames, peak_pos)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(15, 18)); labels = ['early', 'near_peak', 'late']
    frame_by_pos = {int(x['manifest_frame_position']): x for x in per_frame}
    for col, (label, pos) in enumerate(zip(labels, picks)):
        raw = raw_by_position[pos].image.astype(np.float32); corrected = transformed[pos]; meta = frame_by_pos[pos]
        axes[0,col].imshow(raw, cmap='gray'); axes[0,col].set_title(f'{label} raw: pos {pos}')
        axes[1,col].imshow(corrected, cmap='gray'); axes[1,col].set_title(f'{label} corrected')
        axes[2,col].imshow(overlay(corrected, peak)); axes[2,col].set_title(f'corrected/peak overlay\nNCC {meta["raw_ncc_to_peak"]:.3f}→{meta["corrected_ncc_to_peak"]:.3f}')
        for row in range(3): axes[row,col].axis('off')
    x = [f['manifest_frame_position'] for f in frames]; rot=[f['canonical_parameters']['rotation_deg'] for f in frames]; tx=[f['canonical_parameters']['tx'] for f in frames]; ty=[f['canonical_parameters']['ty'] for f in frames]
    axes[3,0].plot(x,rot,'o-'); axes[3,0].axvline(peak_pos,color='k',ls='--'); axes[3,0].set_title('rotation trajectory (deg)')
    axes[3,1].plot(x,tx,'o-',label='tx'); axes[3,1].plot(x,ty,'o-',label='ty'); axes[3,1].axvline(peak_pos,color='k',ls='--'); axes[3,1].legend(); axes[3,1].set_title('translation trajectory (px)')
    axes[3,2].imshow(stable,cmap='gray'); axes[3,2].set_title(f'stable support {stable.mean():.3f}'); axes[3,2].axis('off')
    fig.suptitle(f'{item["group"]}: {item["series_uid"]} {item["phase"]}; max |rotation|={item["max_abs_rotation_deg"]:.2f}°')
    fig.tight_layout(); png=output/f'{item["group"]}__{item["series_uid"]}__{item["phase"]}.png'; fig.savefig(png,dpi=150); plt.close(fig)
    return {key: item[key] for key in ('group','series_uid','split','phase','max_abs_rotation_deg','max_displacement_px')} | {'stable_support_fraction':float(stable.mean()), 'qc_png':str(png), 'frames':per_frame}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-run',required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    root=PROJECT/'outputs'/args.source_run; args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=load_config(PROJECT/'config/default.yaml'); sitkmod=load_v5_module(cfg,'registration_sitk.py')
    selected=choose(root); rows=[make_case(x,args.output_dir,sitkmod) for x in selected]
    pd.DataFrame([{k:v for k,v in row.items() if k!='frames'} for row in rows]).to_csv(args.output_dir/'selected_temporal_nonpeak_qc.csv',index=False)
    atomic_json({'source_run':args.source_run,'method':'saved transforms only; no registration estimation','selected':rows},args.output_dir/'TEMPORAL_NONPEAK_QC.json')
    print(json.dumps({'output':str(args.output_dir),'n':len(rows)},ensure_ascii=False))
if __name__=='__main__': main()
