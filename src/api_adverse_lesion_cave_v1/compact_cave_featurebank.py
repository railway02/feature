#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from common import atomic_json, configure_runtime, load_config, sha256_file, stage_logger, write_marker


RETAINED=("embedding_5120.npy","embedding_views_5120.npz","f4_last_ensemble.fp16.npy","f5_last_ensemble.fp16.npy","phase_trajectories_16.fp16.npz","curves.npz","scalar_features.json","metadata.json","qc.json",".SUCCESS.json")
REMOVABLE_FILES=("probabilities_original.fp16.npz","input_mosaic.jpg","artery_vein_overlay.png","artery_probability.png","vein_probability.png","vessel_probability.png","vessel_union_probability.png")


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--branch",choices=["pred","gt","all_nonzero"],required=True)
    parser.add_argument("--split",choices=["Train","Valid"],required=True)
    args=parser.parse_args(); config=load_config(args.config); configure_runtime(config)
    finish=stage_logger(f"compact_cave_featurebank:{args.branch}:{args.split}")
    outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"]); manifests=Path(config["paths"]["manifests"])
    feature_root=outputs/f"cave_{args.branch}_roi_featurebank"; manifest_path=manifests/f"cave_manifest_{args.branch}_{args.split.casefold()}.csv"
    manifest=pd.read_csv(manifest_path,dtype=str,keep_default_na=False); rows=[]; removed_total=0
    for record in manifest.to_dict("records"):
        for phase in ("pre","post"):
            if str(record.get(f"can_run_{phase}","")).casefold()!="true": continue
            directory=feature_root/args.split.casefold()/str(record["patient_id"])/str(record["series_uid"])/phase
            if not (directory/".SUCCESS.json").is_file(): continue
            missing=[name for name in RETAINED if not (directory/name).is_file()]
            if missing: raise RuntimeError(f"Cannot compact incomplete phase {directory}: {missing}")
            retained_hashes={name:sha256_file(directory/name) for name in RETAINED}
            removed=[]; removed_bytes=0; blocks=directory/"blocks"
            if blocks.exists():
                size=sum(path.stat().st_size for path in blocks.rglob("*") if path.is_file()); shutil.rmtree(blocks); removed.append("blocks/"); removed_bytes+=size
            for name in REMOVABLE_FILES:
                path=directory/name
                if path.is_file(): removed_bytes+=path.stat().st_size; path.unlink(); removed.append(name)
            payload={"status":"compacted","branch":args.branch,"split":args.split,"patient_id":str(record["patient_id"]),"series_uid":str(record["series_uid"]),"phase":phase,"removed":removed,"removed_bytes":removed_bytes,"retained_sha256":retained_hashes}
            atomic_json(payload,directory/".COMPACTED.json"); rows.append(payload); removed_total+=removed_bytes
    summary={"branch":args.branch,"split":args.split,"compacted_phases":len(rows),"removed_bytes":removed_total,"retained_files":list(RETAINED),"manifest_sha256":sha256_file(manifest_path)}
    marker=reports/f".CAVE_{args.branch.upper()}_{args.split.upper()}_COMPACT_SUCCESS"
    write_marker(marker,f"compact_cave_featurebank:{args.branch}:{args.split}",config,{"manifest_sha256":sha256_file(manifest_path)},summary)
    atomic_json(summary,reports/f"cave_{args.branch}_{args.split.casefold()}_compaction.json")
    finish(summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
