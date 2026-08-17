# Complete /2D and CAVE upstream featurebanks v1

This independent package never changes the existing 781/207 outcome banks or the trained main-fusion artifacts. It runs frozen strict SegResNet inference on all 2233 annotated `/2D` phase PNGs and only repackages existing CAVE GTMask-ROI embeddings.

`all2d_phase_features_by_fold.npz` is phase-level `[2233,5,512]` where each 512-D value is `[Global(256), soft PredROI(256)]`. It is not fabricated into 1024-D for single-phase patients. `all2d_prepost_features_by_fold.npz` has only the 1009 patients with both phases and is `[1009,5,1024]`, ordered `[G_pre,PredROI_pre,G_post,PredROI_post]`.

`cave_phase_features.npz` contains the 2209 existing successful GTMask-ROI embeddings `[2209,5120]`; no CAVE encoder is run. `cave_prepost_features.npz` contains the 992 series with both successful phases, `[992,10240]`, Pre then Post. The 413 unavailable source phases are described in `cave_exclusion_manifest.csv`; missing a 2D GT mask means ineligible for this feature version, not a corrupted DSA source.

`complete_2d_cave_prepost_inputs.npz` is the full cross-modal inventory interface: `[992,5,1024]` 2D plus `[992,10240]` CAVE. It uses only `png_key -> segmentation_key` mappings and does not contain labels, averages, or strict outcome OOF semantics.

Run in order:

```bash
PY2D=/root/autodl-tmp/envs/png2d-spatial-v6/bin/python
PYCAVE=/root/autodl-tmp/envs/aneurysm-ml/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_complete_2d_cave_featurebanks_v1
$PY2D $CODE/export_all2d_segresnet_features.py --device cuda:0
$PYCAVE $CODE/consolidate_cave_features.py
$PYCAVE $CODE/build_complete_crossmodal_interface.py
$PYCAVE $CODE/audit_complete_featurebanks.py
```
