from __future__ import annotations

from pathlib import Path
from typing import Iterable
import gzip
import struct

import cv2
import numpy as np

ORIENTATIONS = (
    "identity", "flip_x", "flip_y", "flip_xy",
    "transpose", "transpose_flip_x", "transpose_flip_y", "transpose_flip_xy",
)


def _load_nifti_minimal(path: Path) -> np.ndarray:
    """Minimal NIfTI-1 single-file reader used only when nibabel is unavailable.

    Supports ordinary 2-D/3-D scalar .nii/.nii.gz files. It intentionally does
    not interpret affine orientation; this pipeline derives pixel orientation
    from the paired Image/sequence matching manifest.
    """
    opener = gzip.open if path.name.casefold().endswith(".gz") else open
    with opener(path, "rb") as handle:
        payload = handle.read()
    if len(payload) < 352:
        raise ValueError(f"NIfTI file is too small: {path}")
    little = struct.unpack_from("<i", payload, 0)[0]
    big = struct.unpack_from(">i", payload, 0)[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"Not a NIfTI-1 header (sizeof_hdr != 348): {path}")
    dims = struct.unpack_from(endian + "8h", payload, 40)
    ndim = int(dims[0])
    if ndim < 1 or ndim > 7:
        raise ValueError(f"Invalid NIfTI ndim={ndim}: {path}")
    shape = tuple(int(v) for v in dims[1:1 + ndim])
    datatype = int(struct.unpack_from(endian + "h", payload, 70)[0])
    vox_offset = int(round(float(struct.unpack_from(endian + "f", payload, 108)[0])))
    slope = float(struct.unpack_from(endian + "f", payload, 112)[0])
    inter = float(struct.unpack_from(endian + "f", payload, 116)[0])
    dtype_map = {
        2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
        256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype code {datatype}: {path}")
    dtype = np.dtype(endian + dtype_map[datatype])
    count = int(np.prod(shape, dtype=np.int64))
    if vox_offset < 0 or vox_offset + count * dtype.itemsize > len(payload):
        raise ValueError(f"NIfTI data payload is truncated: {path}")
    array = np.frombuffer(payload, dtype=dtype, count=count, offset=vox_offset).reshape(shape, order="F")
    if np.isfinite(slope) and slope not in (0.0, 1.0):
        array = array.astype(np.float32) * slope
    if np.isfinite(inter) and inter != 0.0:
        array = array.astype(np.float32) + inter
    return np.asarray(array)


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError:  # pragma: no cover - fallback is covered in package tests
        return _load_nifti_minimal(path)
    return np.asanyarray(nib.load(str(path)).dataobj)


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    name = path.name.casefold()
    if name.endswith((".nii", ".nii.gz")):
        return np.asarray(_load_nifti(path))
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(path)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return np.asarray(array)


def _shallow_axis(shape: tuple[int, ...], max_depth: int = 8) -> int:
    candidates = [axis for axis, size in enumerate(shape) if size <= max_depth]
    if not candidates:
        raise ValueError(f"不是二维或浅层二维文件，shape={shape}")
    min_size = min(shape[axis] for axis in candidates)
    best = [axis for axis in candidates if shape[axis] == min_size]
    if len(best) != 1:
        raise ValueError(f"无法唯一确定浅层轴，shape={shape}, candidates={candidates}")
    return best[0]


def reference_planes(array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        return [("2d", array.astype(np.float32, copy=False))]
    if array.ndim != 3:
        raise ValueError(f"参考图必须是二维或浅层三维，shape={array.shape}")
    axis = _shallow_axis(array.shape)
    moved = np.moveaxis(array, axis, 0)
    return [(f"axis{axis}_slice{i}", np.asarray(plane, dtype=np.float32)) for i, plane in enumerate(moved)]


def load_reference_planes(path: str | Path) -> list[tuple[str, np.ndarray]]:
    planes = reference_planes(load_array(path))
    valid: list[tuple[str, np.ndarray]] = []
    for name, plane in planes:
        if plane.ndim != 2 or not np.isfinite(plane).all():
            continue
        if float(np.nanstd(plane)) <= 1e-8:
            continue
        valid.append((name, plane))
    if not valid:
        raise ValueError(f"参考图没有有效二维平面：{path}")
    return valid


def load_label_mask(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    raw = np.asarray(load_array(path))
    raw_shape = tuple(raw.shape)
    array = np.squeeze(raw)
    collapse = "none"
    if array.ndim == 3:
        axis = _shallow_axis(array.shape)
        array = np.max(array, axis=axis)
        collapse = f"max_axis_{axis}"
    if array.ndim != 2:
        raise ValueError(f"Mask 必须是二维或浅层三维，shape={array.shape}, path={path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Mask 存在非有限值：{path}")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-5):
        raise ValueError(f"Mask 不是整数标签图：{path}")
    labels = rounded.astype(np.int32, copy=False)
    return labels, {
        "raw_shape": "x".join(map(str, raw_shape)),
        "collapsed_shape": "x".join(map(str, labels.shape)),
        "collapse_method": collapse,
    }


def apply_orientation(array: np.ndarray, transform: str | None) -> np.ndarray:
    name = str(transform or "identity").strip().casefold().replace("-", "_")
    aliases = {
        "": "identity", "none": "identity", "id": "identity",
        "flipx": "flip_x", "flipy": "flip_y", "flipxy": "flip_xy",
        "transpose_flipx": "transpose_flip_x",
        "transpose_flipy": "transpose_flip_y",
        "transpose_flipxy": "transpose_flip_xy",
    }
    name = aliases.get(name, name)
    if name not in ORIENTATIONS:
        raise ValueError(f"不支持的方向变换：{transform}")
    out = np.asarray(array)
    if name.startswith("transpose"):
        out = out.T
    if name in {"flip_x", "flip_xy", "transpose_flip_x", "transpose_flip_xy"}:
        out = np.fliplr(out)
    if name in {"flip_y", "flip_xy", "transpose_flip_y", "transpose_flip_xy"}:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def resize_labels(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    if mask.shape == (h, w):
        return np.ascontiguousarray(mask)
    result = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.rint(result).astype(np.int32)


def parse_orientations(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ORIENTATIONS
    parsed = tuple(str(value).strip() for value in values)
    unknown = sorted(set(parsed) - set(ORIENTATIONS))
    if unknown:
        raise ValueError(f"未知方向变换：{unknown}")
    return parsed
