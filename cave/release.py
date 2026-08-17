from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import sha256_file, sha256_tree
from cave_model import cave_code_tree_hash, git_commit


def artifact(path: Path, name: str, kind: str = "file") -> dict[str, Any]:
    if kind == "file":
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
    elif kind == "tree":
        if not path.is_dir():
            raise FileNotFoundError(path)
        digest = sha256_tree(path)
    else:
        raise ValueError(kind)
    return {"name": name, "path": str(path.resolve()), "sha256": digest, "kind": kind}


def verify_release(
    release_path: Path,
    required: dict[str, Path],
    cave_repo: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    if not release_path.is_file():
        raise FileNotFoundError(f"Full Valid requires frozen release: {release_path}")
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    lookup = {str(item["name"]): item for item in payload.get("artifacts", [])}
    failures: list[str] = []
    for name, path in required.items():
        item = lookup.get(name)
        if item is None:
            failures.append(f"missing artifact record: {name}")
            continue
        if Path(item["path"]).resolve() != path.resolve():
            failures.append(f"path changed for {name}")
            continue
        actual = sha256_file(path) if item.get("kind") == "file" else sha256_tree(path)
        if actual != item.get("sha256"):
            failures.append(f"SHA changed for {name}")
    if payload.get("cave_commit") != git_commit(cave_repo):
        failures.append("CAVE commit changed")
    if payload.get("cave_code_tree_sha256") != cave_code_tree_hash(cave_repo):
        failures.append("CAVE model code tree changed")
    if payload.get("checkpoint_sha256") != sha256_file(checkpoint):
        failures.append("CAVE checkpoint changed")
    if failures:
        raise AssertionError("Frozen release changed before Valid:\n" + "\n".join(failures))
    return payload
