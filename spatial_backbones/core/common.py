from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    cfg = json.loads(resolved.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(resolved)
    cfg["_config_sha256"] = sha256_file(resolved)
    return cfg


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_text(text: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def atomic_json(value: Any, path: str | Path) -> None:
    atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, target)


def atomic_torch_save(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def tree_manifest(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    if base.is_file():
        return [{"path": base.name, "size": base.stat().st_size, "sha256": sha256_file(base)}]
    rows = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rows.append({
            "path": str(path.relative_to(base)),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return rows


def tree_hash(root: str | Path) -> str:
    return canonical_hash(tree_manifest(root))


def run_signature(cfg: dict[str, Any], split_hash: str, family: str, pretrained_sha256: str) -> dict[str, str]:
    return {
        "config_sha256": cfg["_config_sha256"],
        "split_sha256": split_hash,
        "model_family": family,
        "pretrained_sha256": pretrained_sha256,
    }


def assert_signature(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"Resume signature mismatch for {key}: {actual.get(key)!r} != {value!r}")
