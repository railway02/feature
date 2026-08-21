#!/usr/bin/env python
"""Launch an auditable, resumable cohort run with bounded CPU oversubscription.

This launcher deliberately uses nohup/process groups because tmux is not installed on the
current server.  It never chooses tau or changes registration parameters; callers must
provide a frozen configuration for locked Train/Valid/full runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# Hosted shells may export OMP_NUM_THREADS=0.  Repair it before any optional numerical
# import in metadata collection, then set the requested bounded value for the child.
if not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    files = sorted(list((ROOT / "dsa_reg").glob("*.py")) + list((ROOT / "scripts").glob("*.py")))
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def versions():
    out = {"python": sys.version, "hostname": platform.node()}
    try:
        import SimpleITK as sitk
        out["SimpleITK"] = sitk.Version_VersionString()
    except Exception as e:
        out["SimpleITK"] = repr(e)
    try:
        import ants
        out["ANTsPy"] = getattr(ants, "__version__", "unknown")
    except Exception as e:
        out["ANTsPy"] = repr(e)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Frozen registration YAML")
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-root", required=True, help="New versioned /dsa_registration_runs/<run_id> directory")
    p.add_argument("--split", default=None); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-exploratory-tau", action="store_true")
    a = p.parse_args()
    config, manifest, run = Path(a.config).resolve(), Path(a.manifest).resolve(), Path(a.run_root).resolve()
    if not config.exists() or not manifest.exists():
        raise FileNotFoundError("config or manifest missing")
    if run.exists() and any(run.iterdir()) and not a.resume:
        raise FileExistsError(f"Refusing to overwrite existing run root: {run}; use --resume explicitly")
    run.mkdir(parents=True, exist_ok=True); logs = run / "logs"; logs.mkdir(exist_ok=True)
    resolved_config = run / (
        "PROVISIONAL_REGISTRATION_CONFIG.yaml" if a.allow_exploratory_tau
        else "LOCKED_REGISTRATION_CONFIG.yaml"
    )
    if not resolved_config.exists():
        shutil.copy2(config, resolved_config)
    cfg = yaml.safe_load(resolved_config.read_text())
    output_root = run / "results"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_cohort.py"), "--config", str(resolved_config),
           "--manifest", str(manifest), "--output-root", str(output_root), "--continue-on-error",
           "--workers", str(a.workers)]
    if a.split:
        cmd += ["--split", a.split]
    if a.resume:
        cmd.append("--resume")
    if a.allow_exploratory_tau:
        cmd.append("--allow-exploratory-tau")
    env = os.environ.copy()
    threads = str(max(1, a.threads_per_worker))
    for key in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = threads
    env["PYTHONUNBUFFERED"] = "1"
    metadata = {
        "run_id": run.name, "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": None, "code_hash": code_hash(), "config_hash": sha256_file(resolved_config),
        "manifest_hash": sha256_file(manifest), "manifest": str(manifest), "tau_artifact":
        cfg.get("features", {}).get("expansion_tau_artifact"), "output_root": str(output_root),
        "command": cmd, "workers": a.workers, "threads_per_worker": int(threads),
        "environment_thread_limits": {k: env[k] for k in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "runner": "nohup/process-group", **versions(),
    }
    log = logs / "cohort.log"
    with log.open("ab", buffering=0) as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    metadata["pid"] = proc.pid; metadata["log"] = str(log)
    (run / "RUN_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"pid": proc.pid, "log": str(log), "run_root": str(run), "command": cmd}, indent=2))


if __name__ == "__main__":
    main()
