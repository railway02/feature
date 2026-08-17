from __future__ import annotations

from pathlib import Path
from typing import Any

from common import sha256_json, write_json_atomic
from pooling import AUXILIARY_BLOCKS, PRIMARY_BLOCKS, TRAJECTORY_REGIONS, TRAJECTORY_SCALES, embedding_feature_names

SCHEMA_VERSION = "api_fullseq_cave_v3_featurebank_schema_1"


def build_schema(scalar_names: list[str], frozen_config_hash: str) -> dict[str, Any]:
    embedding_names = embedding_feature_names()
    if len(embedding_names) != 5120 or len(set(embedding_names)) != 5120:
        raise AssertionError("Invalid embedding schema")
    if len(scalar_names) != len(set(scalar_names)):
        raise AssertionError("Duplicate scalar feature names")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frozen_config_hash": frozen_config_hash,
        "embedding_dimension": 5120,
        "embedding_blocks": [
            {
                "name": name,
                "offset_start": block_index * 512,
                "offset_end_exclusive": (block_index + 1) * 512,
                "channels": 512,
            }
            for block_index, name in enumerate(PRIMARY_BLOCKS)
        ],
        "embedding_feature_names": embedding_names,
        "auxiliary_embedding_blocks": list(AUXILIARY_BLOCKS),
        "trajectory_scales": list(TRAJECTORY_SCALES),
        "trajectory_regions": list(TRAJECTORY_REGIONS),
        "scalar_dimension": len(scalar_names),
        "scalar_feature_names": scalar_names,
        "scalar_missing_policy": "nonfinite scientific values are serialized as JSON null and loaded as NaN",
        "qc_is_model_input": False,
    }
    payload["schema_sha256"] = sha256_json(payload)
    return payload


def ensure_schema(path: Path, scalar_names: list[str], frozen_config_hash: str) -> dict[str, Any]:
    expected = build_schema(scalar_names, frozen_config_hash)
    if path.exists():
        import json
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise AssertionError(f"Feature schema differs from existing frozen schema: {path}")
        return current
    write_json_atomic(path, expected)
    return expected
