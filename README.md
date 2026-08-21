# DSA model code

This repository contains only the requested model code:

- `searaft/`: SEA-RAFT optical-flow source and helper scripts.
- `cave/`: CAVE standalone feature-extraction and prediction pipeline code.
- `segresnet/`: SegResNet spatial-model code.
- `deeplabv3/`: DeepLabV3+ spatial-model code.

Excluded: patient data, labels, manifests, outputs, checkpoints, logs, historical reports, release copies, local JSON configurations, and every SEA-RAFT+CAVE fusion/image-probe/key-fusion pipeline.

The CAVE implementation depends on the upstream project: https://github.com/RuishengSu/CAVE_DSA


## DSA registration

- `dsa_registration_local_v1/`: local-crop registration, temporal motion, Jacobian, and hemodynamic code.
- `dsa_registration_fullfov_v5/`: full-field-of-view registration, feature extraction, quality control, and outcome code.

These directories contain source, tests, launch scripts, and dependency metadata only. Local configuration, data, generated outputs, checkpoints, and reports are excluded.
