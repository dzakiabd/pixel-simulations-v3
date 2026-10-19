import numpy as np
import pytest
from PIL import Image

import processing as proc


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


@pytest.fixture
def flat():
    return np.full((32, 32, 3), 90, dtype=np.uint8)


def assert_valid(out, size=(48, 64)):
    assert out.dtype == np.uint8
    assert out.shape[:2] == size


def test_to_rgb_array_handles_modes():
    for mode in ("RGB", "RGBA", "L", "P", "LA"):
        arr = proc.to_rgb_array(Image.new(mode, (10, 8)))
        assert arr.shape == (8, 10, 3) and arr.dtype == np.uint8


def test_to_rgb_array_numpy_inputs():
    assert proc.to_rgb_array(np.zeros((4, 4), np.uint8)).shape == (4, 4, 3)
    assert proc.to_rgb_array(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4, 3)


def test_basic_ops_shape_and_dtype(img):
    for fn in (proc.grayscale, proc.histogram_equalization, proc.contrast_stretching,
               proc.negative, proc.black_and_white, proc.emboss, proc.sepia):
        assert_valid(fn(img))


def test_negative_involution(img):
    assert np.array_equal(proc.negative(proc.negative(img)), img)


def test_contrast_stretching_range_and_flat(img, flat):
    out = proc.contrast_stretching(img)
    assert out.min() == 0 and out.max() == 255
    assert np.array_equal(proc.contrast_stretching(flat), flat)  # tanpa pembagian nol


@pytest.mark.parametrize("value", [-255, -50, 0, 80, 255])
def test_brightness_clipped(img, value):
    out = proc.brightness_adjustment(img, value)
    assert_valid(out)
    if value == 0:
        assert np.array_equal(out, img)


def test_brightness_direction(flat):
    assert proc.brightness_adjustment(flat, 20)[0, 0, 0] == 110
    assert proc.brightness_adjustment(flat, -20)[0, 0, 0] == 70


@pytest.mark.parametrize("ftype", proc.FILTER_TYPES)
@pytest.mark.parametrize("size", [3, 4, 15])
def test_filtering(img, ftype, size):
    assert_valid(proc.image_filtering(img, ftype, size))


def test_filtering_invalid(img):
    with pytest.raises(ValueError):
        proc.image_filtering(img, "Nope", 3)


@pytest.mark.parametrize("ftype", proc.EDGE_TYPES)
def test_edges(img, flat, ftype):
    assert_valid(proc.edge_detection(img, ftype))
    assert proc.edge_detection(flat, ftype).max() == 0 or ftype == "Laplacian"


@pytest.mark.parametrize("ftype", proc.SHARPEN_TYPES)
def test_sharpening(img, flat, ftype):
    assert_valid(proc.image_sharpening(img, ftype, 1.5))
    assert np.array_equal(proc.image_sharpening(flat, ftype, 1.0), flat)
    assert np.array_equal(proc.image_sharpening(img, ftype, 0.0), img)


def test_thresholding(img):
    out = proc.global_thresholding(img, 128)
    assert set(np.unique(out)) <= {0, 255}
    otsu, t = proc.otsu_threshold(img)
    assert set(np.unique(otsu)) <= {0, 255} and 0 <= t <= 255


def test_color_moments_known_values():
    red = np.zeros((10, 10, 3), np.uint8)
    red[:, :, 0] = 255
    m = proc.color_moments(red)
    assert m["Hue"][0] == pytest.approx(0)
    assert m["Saturation"][0] == pytest.approx(255)
    green = np.zeros((10, 10, 3), np.uint8)
    green[:, :, 1] = 255
    assert proc.color_moments(green)["Hue"][0] == pytest.approx(120)


def test_histograms(img):
    gray, hist = proc.gray_histogram(img)
    assert hist.sum() == gray.size
    for h in proc.rgb_histograms(img):
        assert h.shape == (256,) and h.sum() == img.shape[0] * img.shape[1]


def test_resize_methods(img):
    out = proc.resize_with_methods(img, 2.0)
    assert set(out) == {"Nearest Neighbor", "Bilinear", "Bicubic"}
    for arr in out.values():
        assert arr.shape == (96, 128, 3)
    tiny = proc.resize_with_methods(img, 0.001)
    assert all(a.shape[0] >= 1 and a.shape[1] >= 1 for a in tiny.values())


def test_resize_is_capped():
    big = np.zeros((1000, 1000, 3), np.uint8)
    out = proc.resize_with_methods(big, 4.0)
    assert max(next(iter(out.values())).shape[:2]) <= proc.MAX_OUTPUT_SIDE


def test_feature_images(img):
    assert proc.hog_image(img).ndim == 2
    assert_valid(proc.local_binary_pattern_image(img))


def test_sepia_known_value():
    white = np.full((2, 2, 3), 255, np.uint8)
    out = proc.sepia(white)
    assert tuple(out[0, 0]) == (255, 255, 238)


def test_equalization_improves_spread():
    low = np.random.default_rng(1).integers(100, 140, (40, 40, 3), dtype=np.uint8)
    assert proc.histogram_equalization(low).std() > low.std()
