from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_lines(values: Iterable[str]) -> str:
    return sha256_text("\n".join(map(str, values)))


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("inherit_config", None)
    if parent:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        base = json.loads(parent_path.read_text(encoding="utf-8"))
        cfg = deep_merge(base, raw)
    else:
        cfg = raw
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256_file(path)
    return cfg


def configure_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    threads = str(int(runtime.get("omp_num_threads", 8)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    os.environ["PYTHONHASHSEED"] = str(int(runtime.get("seed", 42)))
    visible = runtime.get("cuda_visible_devices", [0])
    if isinstance(visible, list):
        visible = ",".join(map(str, visible))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(visible))


def atomic_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


def atomic_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def normalize_series_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return text


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}


def parse_pipe(value: Any) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def read_single_column_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.is_file():
        return set()
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            frame = pd.read_csv(path, header=None, dtype=str, encoding=encoding, keep_default_na=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")
    values: set[str] = set()
    for value in frame.iloc[:, 0].tolist():
        item = normalize_patient_id(value)
        if not item or item.casefold() == "x" or "病案" in item or item == "序号":
            continue
        values.add(item)
    return values


def write_success(path: str | Path, stage: str, config: dict[str, Any], payload: dict[str, Any]) -> None:
    atomic_json({
        "stage": stage,
        "status": "success",
        "completed_at_utc": utc_now(),
        "config_sha256": config.get("_config_sha256", ""),
        "payload": payload,
    }, path)
