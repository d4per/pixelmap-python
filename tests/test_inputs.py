"""What counts as a photo, and what the caller is told when it is not one."""

import numpy as np
import pixelmap
import pytest
from conftest import shifted_pair, texture


def test_rgb_and_rgba_agree(small_pair):
    rgba_a, rgba_b = small_pair
    rgb = pixelmap.correspond(rgba_a[..., :3], rgba_b[..., :3])
    rgba = pixelmap.correspond(rgba_a, rgba_b)
    # The alpha channel is opaque throughout, so dropping it must change nothing.
    np.testing.assert_array_equal(
        np.isnan(rgb.flow()), np.isnan(rgba.flow()), strict=True
    )
    np.testing.assert_allclose(np.nan_to_num(rgb.flow()), np.nan_to_num(rgba.flow()))


def test_grayscale_is_broadcast_to_three_channels(small_pair):
    a, b = small_pair
    gray_a, gray_b = a[..., 0], b[..., 0]
    assert gray_a.ndim == 2

    two_dim = pixelmap.correspond(gray_a, gray_b)
    three_dim = pixelmap.correspond(gray_a[..., None], gray_b[..., None])
    assert two_dim.coverage > 0.5
    np.testing.assert_allclose(
        np.nan_to_num(two_dim.flow()), np.nan_to_num(three_dim.flow())
    )


def test_non_contiguous_input_is_handled(small_pair):
    a, b = small_pair
    # A reversed view: same pixels, strides NumPy will not hand over as a flat buffer.
    flipped_a, flipped_b = a[:, ::-1], b[:, ::-1]
    assert not flipped_a.flags["C_CONTIGUOUS"]

    mapping = pixelmap.correspond(flipped_a, flipped_b)
    assert mapping.coverage > 0.5
    # Mirroring both photos mirrors the horizontal displacement.
    assert np.nanmean(mapping.flow()[..., 0]) == pytest.approx(5.0, abs=0.5)


def test_pillow_images_work_directly(small_pair):
    Image = pytest.importorskip("PIL.Image")
    a, b = small_pair
    mapping = pixelmap.correspond(
        Image.fromarray(a, mode="RGBA"), Image.fromarray(b, mode="RGBA")
    )
    assert mapping.coverage > 0.5


def test_size_mismatch_names_both_sizes():
    a = texture(200, 150)
    b = texture(200, 160)
    with pytest.raises(pixelmap.SizeMismatchError) as excinfo:
        pixelmap.correspond(a, b)

    assert excinfo.value.first == (200, 150)
    assert excinfo.value.second == (200, 160)
    assert "200x150" in str(excinfo.value)
    # Callers who do not know this hierarchy still catch it.
    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, pixelmap.PixelmapError)


def test_photo_too_small_reports_the_minimum():
    a, b = shifted_pair(20, 20, 1, 1)
    with pytest.raises(pixelmap.PhotoTooSmallError) as excinfo:
        pixelmap.correspond(a, b)

    assert excinfo.value.dimensions == (20, 20)
    assert excinfo.value.minimum == pixelmap.MIN_DIMENSION == 32


def test_empty_photo_is_a_value_error():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pixelmap.correspond(empty, empty)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((40, 40, 3), dtype=np.float32), "uint8"),
        (np.zeros((40, 40, 2), dtype=np.uint8), "channels"),
        (np.zeros((40,), dtype=np.uint8), "shape"),
        (np.zeros((2, 40, 40, 3), dtype=np.uint8), "shape"),
    ],
)
def test_bad_arrays_are_rejected_with_a_useful_message(image, message):
    with pytest.raises(ValueError, match=message):
        pixelmap.correspond(image, image)


def test_the_offending_argument_is_named(small_pair):
    a, _ = small_pair
    with pytest.raises(ValueError, match="photo2"):
        pixelmap.correspond(a, a.astype(np.float32))


def test_unknown_quality_is_rejected(small_pair):
    with pytest.raises(ValueError, match="low"):
        pixelmap.correspond(*small_pair, quality="excellent")
    with pytest.raises(TypeError):
        pixelmap.correspond(*small_pair, quality=2)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_max_round_trip_error_must_be_positive(small_pair, bad):
    with pytest.raises(ValueError, match="positive"):
        pixelmap.correspond(*small_pair, max_round_trip_error=bad)


def test_seed_must_fit_in_64_bits(small_pair):
    with pytest.raises(ValueError, match="64-bit"):
        pixelmap.correspond(*small_pair, seed=-1)
