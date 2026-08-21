from __future__ import annotations
import math
from pathlib import Path
import numpy as np


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK==2.5.6") from e
    return sitk


def as_image(arr: np.ndarray):
    sitk = _sitk()
    return sitk.GetImageFromArray(arr.astype(np.float32))


def as_mask(mask: np.ndarray):
    sitk = _sitk()
    return sitk.GetImageFromArray(mask.astype(np.uint8))


def _initial_transform(kind: str, fixed_img, moving_img):
    sitk = _sitk()
    if kind == "rigid":
        tx = sitk.Euler2DTransform()
    elif kind == "similarity":
        tx = sitk.Similarity2DTransform()
    elif kind == "affine":
        tx = sitk.AffineTransform(2)
    else:
        raise ValueError(kind)
    return sitk.CenteredTransformInitializer(
        fixed_img, moving_img, tx, sitk.CenteredTransformInitializerFilter.GEOMETRY
    )


def register_pair(fixed: np.ndarray, moving: np.ndarray, kind="rigid", fixed_mask=None, moving_mask=None,
                  metric="correlation", shrink_factors=(4,2,1), smoothing_sigmas=(2,1,0),
                  learning_rate=1.0, min_step=1e-3, iterations=160, gradient_tolerance=1e-6,
                  use_moving_mask=True, initial_transform=None):
    sitk = _sitk()
    fi, mi = as_image(fixed), as_image(moving)
    if initial_transform is None:
        tx = _initial_transform(kind, fi, mi)
    elif kind == "rigid":
        tx = sitk.Euler2DTransform(initial_transform)
    elif kind == "similarity":
        tx = sitk.Similarity2DTransform(initial_transform)
    elif kind == "affine":
        tx = sitk.AffineTransform(initial_transform)
    else:
        raise ValueError(kind)

    reg = sitk.ImageRegistrationMethod()
    if metric == "correlation":
        reg.SetMetricAsCorrelation()
    elif metric == "meansquares":
        reg.SetMetricAsMeanSquares()
    elif metric == "mattes":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    else:
        raise ValueError(metric)

    if fixed_mask is not None and np.any(fixed_mask):
        reg.SetMetricFixedMask(as_mask(fixed_mask))
    if use_moving_mask and moving_mask is not None and np.any(moving_mask):
        reg.SetMetricMovingMask(as_mask(moving_mask))

    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate), minStep=float(min_step), numberOfIterations=int(iterations),
        gradientMagnitudeTolerance=float(gradient_tolerance), relaxationFactor=0.5
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([int(x) for x in shrink_factors])
    reg.SetSmoothingSigmasPerLevel([float(x) for x in smoothing_sigmas])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fi, mi)
    return tx, {
        "metric_value": float(reg.GetMetricValue()),
        "optimizer_iteration": int(reg.GetOptimizerIteration()),
        "stop_condition": str(reg.GetOptimizerStopConditionDescription()),
    }


def resample(moving: np.ndarray, fixed_shape_source: np.ndarray, tx, is_mask=False, default=0.0) -> np.ndarray:
    sitk = _sitk()
    fixed_img = as_image(fixed_shape_source)
    moving_img = as_mask(moving) if is_mask else as_image(moving)
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    out = sitk.Resample(moving_img, fixed_img, tx, interp, float(default), moving_img.GetPixelID())
    arr = sitk.GetArrayFromImage(out)
    return arr.astype(bool) if is_mask else arr.astype(np.float32)


def save_transform(tx, path: str | Path):
    sitk = _sitk()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(path))


def interpolate_rigid_transforms(left, right, alpha: float):
    """Interpolate two fixed->moving Euler2D transforms for low-contrast frames."""
    sitk = _sitk()
    if left is None and right is None:
        return None
    template = left if left is not None else right
    center = tuple(float(x) for x in sitk.Euler2DTransform(template).GetCenter())

    def params(tx):
        if tx is None:
            return 0.0, np.zeros(2, dtype=float)
        t = sitk.Euler2DTransform(tx)
        return float(t.GetAngle()), np.asarray(t.GetTranslation(), dtype=float)

    a0, p0 = params(left); a1, p1 = params(right)
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    w = float(np.clip(alpha, 0.0, 1.0))
    out = sitk.Euler2DTransform()
    out.SetCenter(center)
    out.SetAngle(a0 + w * da)
    out.SetTranslation(tuple(p0 + w * (p1 - p0)))
    return out


def canonical_parameters(tx, kind: str) -> dict:
    """Return moving->fixed (Pre->Post) parameters in the explicit y=A x+b form.

    SimpleITK resampling transforms map output/fixed points to input/moving points.
    The registration transform is therefore inverted before reporting the project's
    canonical Pre->Post direction. For centered transforms, GetTranslation() is a
    parameter around the transform center and is NOT the same as the t in y=R x+t.
    We report the effective offset b = T(0) as tx/ty, and keep the centered parameter
    translation separately for QC.
    """
    sitk = _sitk()
    inv = tx.GetInverse()

    def effective_offset(t):
        p0 = t.TransformPoint((0.0, 0.0))
        return float(p0[0]), float(p0[1])

    if kind == "rigid":
        t = sitk.Euler2DTransform(inv)
        bx, by = effective_offset(t)
        return {
            "rotation_rad": float(t.GetAngle()),
            "rotation_deg": float(np.degrees(t.GetAngle())),
            "tx": bx,
            "ty": by,
            "parameter_tx": float(t.GetTranslation()[0]),
            "parameter_ty": float(t.GetTranslation()[1]),
            "center_x": float(t.GetCenter()[0]),
            "center_y": float(t.GetCenter()[1]),
            "scale": 1.0,
        }
    if kind == "similarity":
        t = sitk.Similarity2DTransform(inv)
        bx, by = effective_offset(t)
        return {
            "rotation_rad": float(t.GetAngle()),
            "rotation_deg": float(np.degrees(t.GetAngle())),
            "tx": bx,
            "ty": by,
            "parameter_tx": float(t.GetTranslation()[0]),
            "parameter_ty": float(t.GetTranslation()[1]),
            "center_x": float(t.GetCenter()[0]),
            "center_y": float(t.GetCenter()[1]),
            "scale": float(t.GetScale()),
        }
    if kind == "affine":
        t = sitk.AffineTransform(inv)
        A = np.asarray(t.GetMatrix(), dtype=float).reshape(2, 2)
        u, s, vt = np.linalg.svd(A)
        R = u @ vt
        angle = math.atan2(R[1, 0], R[0, 0])
        bx, by = effective_offset(t)
        return {
            "rotation_rad": float(angle),
            "rotation_deg": float(np.degrees(angle)),
            "tx": bx,
            "ty": by,
            "parameter_tx": float(t.GetTranslation()[0]),
            "parameter_ty": float(t.GetTranslation()[1]),
            "center_x": float(t.GetCenter()[0]),
            "center_y": float(t.GetCenter()[1]),
            "scale_x_svd": float(s[0]),
            "scale_y_svd": float(s[1]),
            "affine_a00": float(A[0, 0]), "affine_a01": float(A[0, 1]),
            "affine_a10": float(A[1, 0]), "affine_a11": float(A[1, 1]),
        }
    raise ValueError(kind)
