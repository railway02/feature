#!/usr/bin/env python
"""Validate runtime dependencies and the API surface used by this project."""
import inspect, sys
mods=['numpy','pandas','scipy','skimage','cv2','yaml','matplotlib']
for name in mods:
    m=__import__(name); print(f'{name}: {getattr(m,"__version__","ok")}')
try:
    import SimpleITK as sitk
    print('SimpleITK:',sitk.Version_VersionString())
    assert hasattr(sitk.ImageRegistrationMethod(),'SetMetricMovingMask')
    assert hasattr(sitk,'Similarity2DTransform')
except Exception as e:
    print('FAIL SimpleITK:',repr(e)); sys.exit(2)
try:
    import ants
    print('ANTsPy:',getattr(ants,'__version__','unknown'))
    sig=inspect.signature(ants.registration)
    for p in ['fixed','moving','type_of_transform','initial_transform','mask','moving_mask','mask_all_stages','syn_metric','syn_sampling','reg_iterations']:
        assert p in sig.parameters,p
    sig2=inspect.signature(ants.create_jacobian_determinant_image)
    for p in ['domain_image','tx','do_log','geom']:
        assert p in sig2.parameters,p
    sig3=inspect.signature(ants.apply_transforms)
    for p in ['fixed','moving','transformlist','interpolator','defaultvalue']:
        assert p in sig3.parameters,p
except Exception as e:
    print('FAIL ANTsPy:',repr(e)); sys.exit(3)
print('API VALIDATION PASS')
