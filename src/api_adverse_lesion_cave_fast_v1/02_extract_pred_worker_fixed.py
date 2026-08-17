#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import fast_roi


def main() -> int:
    source = Path(__file__).with_name("02_extract_pred_worker.py")
    spec = importlib.util.spec_from_file_location("fast_v1_pred_worker_impl", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.import_roi_helpers = lambda _path: fast_roi
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
