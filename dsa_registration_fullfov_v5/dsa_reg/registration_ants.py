from __future__ import annotations
from pathlib import Path
import numpy as np


def _ants():
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "ANTsPy is required for deformable registration. Install antspyx==0.6.3 "
            "in the same environment used to run this pipeline."
        ) from e
    return ants


def _aimg(arr: np.ndarray):
    ants = _ants()
    return ants.from_numpy(np.asarray(arr, dtype=np.float32))


def _amask(mask: np.ndarray):
    ants = _ants()
    return ants.from_numpy(np.asarray(mask, dtype=np.uint8))


def _copy_scalar_like(arr: np.ndarray, ref_img):
    ants = _ants()
    return ants.from_numpy(
        np.asarray(arr, dtype=np.float32),
        origin=ref_img.origin,
        spacing=ref_img.spacing,
        direction=ref_img.direction,
    )


def _find_warp(paths, inverse=False):
    for p in paths:
        name = Path(p).name
        if inverse and "InverseWarp" in name:
            return p
        if (not inverse) and "Warp" in name and "InverseWarp" not in name:
            return p
    return None


def _assert_residual_linear_identity(transform_paths, atol=1e-6):
    """SyNOnly emits an identity .mat in ANTsPy 0.6.3; reject any non-identity one."""
    ants = _ants()
    checked = []
    for path in transform_paths:
        if not str(path).endswith(".mat"):
            continue
        tx = ants.read_transform(str(path))
        params = np.asarray(tx.parameters, dtype=float)
        if params.size < 6:
            raise RuntimeError(f"Unexpected residual linear transform parameters in {path}: {params}")
        A = params[:4].reshape(2, 2)
        t = params[4:6]
        if not np.allclose(A, np.eye(2), atol=atol, rtol=0) or not np.allclose(t, 0, atol=atol, rtol=0):
            raise RuntimeError(
                f"Non-identity linear transform found in SyN residual chain: {path}, A={A}, t={t}. "
                "Global scale/affine must not enter the disease Jacobian."
            )
        checked.append(str(path))
    return checked


def _warp_scalar_to_fixed(arr_moving: np.ndarray, moving_ref, fixed_ref, fwdtransforms,
                          interpolator="linear", defaultvalue=0.0) -> np.ndarray:
    ants = _ants()
    img = _copy_scalar_like(arr_moving, moving_ref)
    out = ants.apply_transforms(
        fixed=fixed_ref,
        moving=img,
        transformlist=fwdtransforms,
        interpolator=interpolator,
        defaultvalue=float(defaultvalue),
        singleprecision=True,
    )
    return out.numpy().astype(np.float32)


def _warp_fixed_scalar_to_moving(arr_fixed: np.ndarray, fixed_ref, moving_ref, invtransforms,
                                 interpolator="linear", defaultvalue=0.0) -> np.ndarray:
    ants = _ants()
    img = _copy_scalar_like(arr_fixed, fixed_ref)
    out = ants.apply_transforms(
        fixed=moving_ref,
        moving=img,
        transformlist=invtransforms,
        interpolator=interpolator,
        defaultvalue=float(defaultvalue),
        singleprecision=True,
    )
    return out.numpy().astype(np.float32)


def warp_mask_to_fixed(mask_moving: np.ndarray, fixed: np.ndarray, moving: np.ndarray, fwdtransforms) -> np.ndarray:
    ants = _ants()
    fi, mi = _aimg(fixed), _aimg(moving)
    mask_img = ants.from_numpy(
        np.asarray(mask_moving, dtype=np.uint8),
        origin=mi.origin,
        spacing=mi.spacing,
        direction=mi.direction,
    )
    out = ants.apply_transforms(
        fixed=fi,
        moving=mask_img,
        transformlist=fwdtransforms,
        interpolator="genericLabel",
        defaultvalue=0,
        singleprecision=True,
    )
    return out.numpy() > 0.5


def _canonical_inverse_warp_jacobian(mi, fi, inv_warp: str, fwd_warp: str,
                                     fwdtransforms, invtransforms, geom: bool):
    """Compute canonical moving->fixed Jacobian from the inverse warp itself.

    ANTs image resampling uses the so-called forward warp as a fixed->moving point map.
    The project's clinical convention is the opposite point direction: moving (Pre-global)
    -> fixed (Post). Therefore we compute the determinant directly from InverseWarp in its
    native moving-space domain, rather than inferring it by negating/reciprocating the
    forward-warp determinant at the wrong spatial location.

    The scalar log-Jacobian is then resampled into fixed/Post space for regional statistics
    and visualization. This preserves the requested convention:
        logJ > 0  => Pre->Post local area expansion
        logJ < 0  => Pre->Post local area contraction
    """
    ants = _ants()

    # Canonical Pre-global -> Post determinant, natively defined on moving space.
    jac_m_img = ants.create_jacobian_determinant_image(mi, inv_warp, do_log=False, geom=geom)
    jac_m = jac_m_img.numpy().astype(np.float32)
    valid_m = np.isfinite(jac_m) & (jac_m > 0)
    logj_m = np.full_like(jac_m, np.nan, dtype=np.float32)
    logj_m[valid_m] = np.log(jac_m[valid_m])
    fold_m = np.isfinite(jac_m) & (jac_m <= 0)

    # Resample logJ and support to fixed/Post space. Interpolate logJ, not J, so that
    # expansion/contraction remain symmetric around 0 after interpolation.
    log_safe = np.where(valid_m, logj_m, 0.0).astype(np.float32)
    logj_post = _warp_scalar_to_fixed(log_safe, mi, fi, fwdtransforms, "linear", 0.0)
    valid_post = _warp_scalar_to_fixed(valid_m.astype(np.float32), mi, fi, fwdtransforms,
                                       "nearestNeighbor", 0.0) > 0.5
    fold_post = _warp_scalar_to_fixed(fold_m.astype(np.float32), mi, fi, fwdtransforms,
                                      "nearestNeighbor", 0.0) > 0.5
    logj_post[~valid_post] = np.nan
    jac_post = np.full_like(logj_post, np.nan, dtype=np.float32)
    jac_post[valid_post] = np.exp(logj_post[valid_post])

    # Forward-pull determinant is retained only for transform-direction QC.
    fwd_j_img = ants.create_jacobian_determinant_image(fi, fwd_warp, do_log=False, geom=geom)
    fwd_j = fwd_j_img.numpy().astype(np.float32)
    fwd_valid = np.isfinite(fwd_j) & (fwd_j > 0)
    fwd_log = np.full_like(fwd_j, np.nan, dtype=np.float32)
    fwd_log[fwd_valid] = np.log(fwd_j[fwd_valid])

    # Inverse consistency identity: log det(D phi^{-1})(x) = -log det(D phi)(phi^{-1}(x)).
    # Bring forward logJ into moving space, then compare there.
    fwd_safe = np.where(fwd_valid, fwd_log, 0.0).astype(np.float32)
    fwd_log_m = _warp_fixed_scalar_to_moving(fwd_safe, fi, mi, invtransforms, "linear", 0.0)
    fwd_valid_m = _warp_fixed_scalar_to_moving(fwd_valid.astype(np.float32), fi, mi, invtransforms,
                                               "nearestNeighbor", 0.0) > 0.5
    ic_mask = valid_m & fwd_valid_m & np.isfinite(fwd_log_m)
    if np.any(ic_mask):
        inverse_consistency_logjac_mae = float(np.median(np.abs(logj_m[ic_mask] + fwd_log_m[ic_mask])))
    else:
        inverse_consistency_logjac_mae = np.nan

    return {
        "canonical_jac_moving": jac_m,
        "canonical_logjac_moving": logj_m,
        "canonical_valid_moving": valid_m,
        "canonical_folding_moving": fold_m,
        "canonical_jac": jac_post,
        "canonical_logjac": logj_post,
        "canonical_valid": valid_post,
        "canonical_folding": fold_post,
        "forward_pull_jac": fwd_j,
        "inverse_consistency_logjac_mae": inverse_consistency_logjac_mae,
    }


def run_syn_residual(fixed: np.ndarray, moving_global: np.ndarray,
                     fixed_anchor: np.ndarray | None, moving_anchor: np.ndarray | None,
                     outprefix: str, cfg: dict,
                     fixed_lesion: np.ndarray | None = None,
                     moving_lesion: np.ndarray | None = None):
    ants = _ants()
    fi, mi = _aimg(fixed), _aimg(moving_global)
    fmask = _amask(fixed_anchor) if fixed_anchor is not None and np.any(fixed_anchor) else None
    mmask = _amask(moving_anchor) if moving_anchor is not None and np.any(moving_anchor) else None
    lesion_weight = float(cfg.get("lesion_metric_weight", 0.0))
    multivariate_extras = None
    if (lesion_weight > 0 and fixed_lesion is not None and moving_lesion is not None
            and np.any(fixed_lesion) and np.any(moving_lesion)):
        fixed_lesion_img = _copy_scalar_like(np.asarray(fixed_lesion, dtype=np.float32), fi)
        moving_lesion_img = _copy_scalar_like(np.asarray(moving_lesion, dtype=np.float32), mi)
        multivariate_extras = [(
            str(cfg.get("lesion_metric", "MeanSquares")),
            fixed_lesion_img,
            moving_lesion_img,
            lesion_weight,
            int(cfg.get("lesion_metric_sampling", 0)),
        )]
    Path(outprefix).parent.mkdir(parents=True, exist_ok=True)

    reg = ants.registration(
        fixed=fi,
        moving=mi,
        type_of_transform=cfg.get("transform", "SyNOnly"),
        initial_transform="Identity",
        outprefix=outprefix,
        mask=fmask,
        moving_mask=mmask,
        mask_all_stages=True,
        syn_metric=cfg.get("metric", "CC"),
        syn_sampling=int(cfg.get("syn_sampling", 2)),
        multivariate_extras=multivariate_extras,
        reg_iterations=tuple(int(x) for x in cfg.get("reg_iterations", [60, 40, 20])),
        grad_step=float(cfg.get("grad_step", 0.10)),
        flow_sigma=float(cfg.get("flow_sigma", 3)),
        total_sigma=float(cfg.get("total_sigma", 1)),
        singleprecision=bool(cfg.get("singleprecision", True)),
        verbose=bool(cfg.get("verbose", False)),
    )

    fwd_warp = _find_warp(reg["fwdtransforms"], inverse=False)
    inv_warp = _find_warp(reg["invtransforms"], inverse=True)
    residual_identity_linears = _assert_residual_linear_identity(
        list(dict.fromkeys(list(reg["fwdtransforms"]) + list(reg["invtransforms"])))
    )
    if fwd_warp is None or inv_warp is None:
        raise RuntimeError(
            "SyNOnly did not return both Warp and InverseWarp; cannot compute a canonical "
            f"Pre->Post residual field. fwd={reg['fwdtransforms']} inv={reg['invtransforms']}"
        )

    geom = bool(cfg.get("use_geometric_jacobian", True))
    jac = _canonical_inverse_warp_jacobian(
        mi=mi,
        fi=fi,
        inv_warp=inv_warp,
        fwd_warp=fwd_warp,
        fwdtransforms=reg["fwdtransforms"],
        invtransforms=reg["invtransforms"],
        geom=geom,
    )

    # Canonical Pre-global -> Post displacement magnitude from InverseWarp, first
    # in its native moving space, then resampled to Post space.
    inv_field_img = ants.image_read(inv_warp)
    inv_field = inv_field_img.numpy()
    if inv_field.ndim < 3 or inv_field.shape[-1] != 2:
        raise RuntimeError(f"Expected a 2-D vector displacement field, got shape {inv_field.shape}")
    disp_moving = np.linalg.norm(inv_field, axis=-1).astype(np.float32)
    disp_post = _warp_scalar_to_fixed(disp_moving, mi, fi, reg["fwdtransforms"], "linear", 0.0)
    valid_post = jac["canonical_valid"]
    disp_post[~valid_post] = np.nan

    return {
        "warped_moving": reg["warpedmovout"].numpy().astype(np.float32),
        "warped_fixed": reg["warpedfixout"].numpy().astype(np.float32),
        "fwdtransforms": list(reg["fwdtransforms"]),
        "invtransforms": list(reg["invtransforms"]),
        "fwd_warp": fwd_warp,
        "inv_warp": inv_warp,
        "lesion_metric_enabled": bool(multivariate_extras),
        "lesion_metric_weight": lesion_weight if multivariate_extras else 0.0,
        "residual_identity_linear_transforms": residual_identity_linears,
        "displacement_moving": disp_moving,
        "displacement": disp_post,
        **jac,
    }
