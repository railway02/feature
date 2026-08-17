from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        sanitize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, suffixes: tuple[str, ...] = (".py", ".json", ".sh", ".md")) -> str:
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes):
        rows.append(f"{path.relative_to(root)}\t{sha256_file(path)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def hash_lines(values: list[str]) -> str:
    material = "\n".join(str(v) for v in values).encode("utf-8")
    return hashlib.sha256(material).hexdigest() if values else ""


def parse_pipe_strings(value: Any) -> list[str]:
    if value is None or pd.isna(value) or str(value).strip().casefold() in {"", "nan", "none"}:
        return []
    return [part for part in str(value).split("|") if part != ""]


def parse_pipe_ints(value: Any) -> list[int]:
    return [int(part) for part in parse_pipe_strings(value)]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


@contextmanager
def atomic_directory(final_dir: Path, overwrite: bool = False) -> Iterator[Path]:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_dir.with_name(f".{final_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    backup: Path | None = None
    try:
        yield temporary
        if final_dir.exists():
            if not overwrite:
                raise FileExistsError(final_dir)
            backup = final_dir.with_name(f".{final_dir.name}.{uuid.uuid4().hex}.old")
            os.replace(final_dir, backup)
        os.replace(temporary, final_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not final_dir.exists():
            os.replace(backup, final_dir)
        raise


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
