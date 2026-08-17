#!/usr/bin/env python3
"""Run the resumable ROI Pilot visual batch, then its finalizer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/aneurysm")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    os.chdir(root)
    atomic_write(args.pid_file, f"{os.getpid()}\n")

    auth_path = Path.home() / ".codex" / "auth.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))["OPENAI_API_KEY"]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = auth
    auth = "REDACTED"

    state = {
        "pid": os.getpid(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "visual_batch",
        "visual_batch_exit_code": None,
        "finalizer_exit_code": None,
        "finished_utc": None,
    }
    atomic_write(args.state_file, json.dumps(state, indent=2) + "\n")

    visual_command = [
        sys.executable,
        "code/31_call_roi_pilot_visual_model.py",
        "--model",
        "gpt-5.6-sol",
        "--base-url",
        "http://127.0.0.1:18317/v1",
        "--workers",
        "1",
        "--max-retries",
        "4",
        "--timeout-seconds",
        "300",
        "--resume",
    ]
    visual = subprocess.run(visual_command, env=env, check=False)
    state["visual_batch_exit_code"] = visual.returncode
    if visual.returncode != 0:
        state["stage"] = "visual_batch_failed"
        state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write(args.state_file, json.dumps(state, indent=2) + "\n")
        return visual.returncode

    state["stage"] = "finalizer"
    atomic_write(args.state_file, json.dumps(state, indent=2) + "\n")
    finalizer = subprocess.run(
        [sys.executable, "code/32_finalize_roi_pilot_ai_review.py"],
        env=env,
        check=False,
    )
    state["finalizer_exit_code"] = finalizer.returncode
    state["stage"] = "complete" if finalizer.returncode == 0 else "finalizer_failed"
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write(args.state_file, json.dumps(state, indent=2) + "\n")
    return finalizer.returncode


if __name__ == "__main__":
    raise SystemExit(main())
