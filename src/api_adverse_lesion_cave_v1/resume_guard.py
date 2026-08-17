#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config, marker_matches, sha256_file


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--marker",required=True)
    parser.add_argument("--input-file",action="append",default=[])
    parser.add_argument("--required",action="append",default=[])
    args=parser.parse_args()
    config=load_config(args.config)
    required=[Path(value) for value in args.required]
    current=marker_matches(Path(args.marker),config,required)
    if current and args.input_file:
        payload=json.loads(Path(args.marker).read_text(encoding="utf-8"))
        for specification in args.input_file:
            key,value=specification.split("=",1); path=Path(value)
            if not path.is_file():
                current=False; break
            actual=sha256_file(path)
            if payload.get("inputs",{}).get(key)!=actual:
                current=False; break
    print(f"[RESUME {'HIT' if current else 'MISS'}] {args.marker}")
    return 0 if current else 1


if __name__=="__main__": raise SystemExit(main())
