from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_CONFIG = Path("/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_cave_fast_v1.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    item = Path(path or DEFAULT_CONFIG).resolve()
    config = json.loads(item.read_text(encoding="utf-8"))
    config["_config_path"] = str(item)
    config["_config_sha256"] = sha256_file(item)
    return config


def configure_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    threads = str(int(runtime.get("omp_num_threads", 8)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    os.environ["PYTHONHASHSEED"] = str(int(runtime.get("seed", 42)))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(runtime.get("cuda_visible_devices", "0")))


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_lines(values: Iterable[str]) -> str:
    return sha256_text("\n".join(map(str, values)))


def atomic_json(payload: Any, path: str | Path) -> None:
    item = Path(path); item.parent.mkdir(parents=True, exist_ok=True)
    temporary = item.with_name(f".{item.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, item)


def atomic_text(value: str, path: str | Path) -> None:
    item = Path(path); item.parent.mkdir(parents=True, exist_ok=True)
    temporary = item.with_name(f".{item.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, item)


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    item = Path(path); item.parent.mkdir(parents=True, exist_ok=True)
    temporary = item.with_name(f".{item.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, item)


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def file_evidence(path: str | Path) -> dict[str, Any]:
    item = Path(path).resolve()
    if not item.is_file():
        raise FileNotFoundError(item)
    return {
        "path": str(item),
        "sha256": sha256_file(item),
        "size_bytes": item.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(item.stat().st_mtime, timezone.utc).isoformat(),
    }


def csv_evidence(path: str | Path) -> dict[str, Any]:
    evidence = file_evidence(path)
    evidence["rows"] = int(len(pd.read_csv(path, dtype=str, keep_default_na=False)))
    return evidence


def run(command: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def process_rows(pattern: str) -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid,ppid,pgid,sid,stat,etime,args"], text=True)
    rows = []
    for line in output.splitlines()[1:]:
        if pattern not in line:
            continue
        parts = line.strip().split(None, 6)
        if len(parts) == 7:
            rows.append(dict(zip(("pid", "ppid", "pgid", "sid", "stat", "etime", "args"), parts)))
    return rows


def write_success(path: str | Path, stage: str, config: dict[str, Any], payload: dict[str, Any]) -> None:
    atomic_json({
        "stage": stage,
        "status": "success",
        "completed_at_utc": utc_now(),
        "config_sha256": config["_config_sha256"],
        "payload": payload,
    }, path)
