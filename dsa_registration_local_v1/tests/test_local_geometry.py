import numpy as np

from dsa_local_reg.local_geometry import BBox, crop_with_border_median_padding, resize_whole_canvas, scale_bbox


def test_independent_crop_retains_each_phase_size():
    pre = np.zeros((100, 100), dtype=np.uint8)
    post = np.zeros((140, 120), dtype=np.uint8)
    pre_crop = crop_with_border_median_padding(pre, BBox(10, 20, 70, 80))
    post_crop = crop_with_border_median_padding(post, BBox(5, 6, 105, 130))
    assert pre_crop.image.shape == (60, 60)
    assert post_crop.image.shape == (124, 100)
    assert pre_crop.image.shape != post_crop.image.shape


def test_g1_scales_whole_canvas_and_bbox_together():
    source = np.zeros((80, 120), dtype=np.uint8)
    box = BBox(20, 10, 100, 70)
    resized = resize_whole_canvas(source, (160, 240))
    mapped = scale_bbox(box, source.shape, resized.shape)
    assert mapped == BBox(40, 20, 200, 140)


def test_padding_support_excludes_padded_pixels():
    image = np.ones((10, 10), dtype=np.uint8)
    result = crop_with_border_median_padding(image, BBox(-2, -1, 4, 5))
    assert result.image.shape == (6, 6)
    assert result.valid_support.sum() == 20
