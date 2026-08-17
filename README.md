# DSA aneurysm feature-extraction and prediction pipeline

This repository is a source-only, reproducible-code snapshot of the DSA aneurysm project. It covers data inventory and manifest construction, optical-flow feature extraction (SEA-RAFT), CAVE/ROI feature pipelines, 2D spatial segmentation features, and outcome-prediction experiments.

No patient images, spreadsheets, manifests, model checkpoints, generated features, logs, or experiment outputs are included. The repository should be kept private unless the underlying study data and reports have been approved for public release.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Project scripts and reusable modules, retaining their original filenames and module layout. |
| `src/SEA-RAFT/` | Local SEA-RAFT source snapshot used by the optical-flow scripts. |
| `src/teacher_gtroi/` | GTROI gated teacher extraction and fusion-training source. |
| `configs/` | Versioned pipeline configurations; update their absolute local paths before running elsewhere. |
| `scripts/` | Existing top-level launch scripts. |
| `docs/teacher-gtroi/` | Teacher-model architecture and audit documentation. |

## Getting started

Use Python 3.10+ and install dependencies in an isolated environment. PyTorch must match the target CUDA driver; install the appropriate PyTorch wheel first, then install the remaining packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Before running a pipeline, copy or adapt its configuration under `configs/` to point to your local, access-controlled input and output locations. Do not commit those local overrides, source data, generated feature banks, or checkpoints.

Many scripts are designed to be invoked from their own directory because they use sibling imports. For example:

```bash
cd src/api_fullseq_v3
python prepare_manifests.py --help
```

## Important notes

- The code reflects several sequential research pipelines. Keep the selected config, manifest version, and source directory layout consistent for a run.
- `src/31_call_roi_pilot_visual_model.py` requires `OPENAI_API_KEY` in the environment. It does not contain or read a committed key.
- CAVE-related scripts depend on locally installed model/source dependencies that are deliberately not vendored here; configure these paths in the relevant JSON configuration.
- Historical patch copies (`*.orig`, `*.rej`, `*.bak*`) and generated artifacts are excluded from version control.

## Data policy

This repository intentionally contains code and non-sensitive configuration only. Clinical images, patient-level tables, labels, derived manifests, feature arrays, trained weights, and runtime logs must remain in access-controlled storage.
