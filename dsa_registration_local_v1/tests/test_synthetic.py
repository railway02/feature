import numpy as np

from dsa_local_reg.synthetic import (
    jacobian_det_from_displacement_yx,
    local_radial_field,
    run_synthetic_suite,
    translation_field,
)


def test_translation_has_unit_jacobian():
    jac = jacobian_det_from_displacement_yx(translation_field((41, 43), tx=8, ty=-3))
    assert np.allclose(jac, 1.0)


def test_local_expansion_and_contraction_have_opposite_logjac_sign():
    shape = (81, 83)
    center = (shape[0] // 2, shape[1] // 2)
    expansion = jacobian_det_from_displacement_yx(local_radial_field(shape, 0.08, 15.0))
    contraction = jacobian_det_from_displacement_yx(local_radial_field(shape, -0.08, 15.0))
    assert np.log(expansion[center]) > 0
    assert np.log(contraction[center]) < 0


def test_full_synthetic_suite_passes():
    assert all(item["passed"] for item in run_synthetic_suite())
