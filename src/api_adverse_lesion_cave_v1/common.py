from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_CONFIG = Path("/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_cave_v1.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def hash_lines(values: Iterable[str]) -> str:
    return sha256_text("\n".join(str(item) for item in values))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("ADVERSE_LESION_CONFIG", DEFAULT_CONFIG)).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = sha256_file(config_path)
    return config


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


def pipeline_code_sha256(config: dict[str,Any]) -> str:
    root=Path(config["paths"]["code"])
    files=sorted([path for path in root.rglob("*") if path.is_file() and path.suffix in {".py",".sh"}])
    payload=[f"{path.relative_to(root)}:{sha256_file(path)}" for path in files]
    return hash_lines(payload)


def marker_payload(stage: str, config: dict[str, Any], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "success",
        "completed_at_utc": utc_now(),
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "inputs": inputs,
        "pipeline_code_sha256":pipeline_code_sha256(config),
        "outputs": outputs,
    }


def write_marker(path: Path, stage: str, config: dict[str, Any], inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    atomic_json(marker_payload(stage, config, inputs, outputs), path)


def marker_matches(path: Path, config: dict[str, Any], required_outputs: Iterable[Path] = ()) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("status") == "success"
        and payload.get("config_sha256") == config["_config_sha256"]
        and payload.get("pipeline_code_sha256")==pipeline_code_sha256(config)
        and all(item.exists() for item in required_outputs)
    )


def quarantine(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.quarantine_{stamp}_{reason}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.quarantine_{stamp}_{reason}_{counter}")
        counter += 1
    shutil.move(str(path), str(target))
    return target


def configure_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    threads = str(int(runtime.get("omp_num_threads", 8)))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["PYTHONHASHSEED"] = str(int(runtime.get("seed", 42)))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(runtime.get("cuda_visible_devices", "0")))


def run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(f"[RUN] {printable}", flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def stage_logger(stage: str):
    start = time.time()
    print(f"[STAGE START] {stage} {utc_now()}", flush=True)

    def finish(extra: dict[str, Any] | None = None) -> None:
        payload = {"stage": stage, "elapsed_seconds": time.time() - start, **(extra or {})}
        print("[STAGE PASS] " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

    return finish


def require_file(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    item = Path(path).resolve()
    if not item.is_file():
        raise FileNotFoundError(item)
    actual = sha256_file(item) if expected_sha256 else None
    if expected_sha256 and actual != expected_sha256:
        raise AssertionError(f"SHA256 mismatch: {item} expected={expected_sha256} actual={actual}")
    return {"path": str(item), "size_bytes": item.stat().st_size, "sha256": actual}


def parse_pipe(value: Any) -> list[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return []
    return [item for item in str(value).split("|") if item]


def bool_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def safe_uid(*parts: Any) -> str:
    canonical = "|".join(str(part) for part in parts)
    return sha256_text(canonical)[:24]


def import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)
