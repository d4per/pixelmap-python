"""Point lookup, and the warped image the mapping can produce."""

import numpy as np
import pixelmap
import pytest


@pytest.fixture(scope="module")
def mapping(pair):
    return pixelmap.correspond(*pair)


def test_scalar_lookup_returns_a_tuple_or_none(mapping):
    result = mapping.lookup(200.0, 150.0)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)

    # A point well inside a densely textured image should be mapped.
    x, y = mapping.lookup(200.0, 150.0)
    assert x == pytest.approx(200.0 - 6.0, abs=1.5)
    assert y == pytest.approx(150.0 - 3.0, abs=1.5)


def test_lookup_outside_the_photo_is_unmapped(mapping):
    assert mapping.lookup(-50.0, -50.0) is None
    assert mapping.lookup(10_000.0, 10_000.0) is None


def test_array_lookup_preserves_shape(mapping):
    xs = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    ys = np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float32)

    out_x, out_y = mapping.lookup(xs, ys)
    assert out_x.shape == out_y.shape == (2, 2)
    assert out_x.dtype == np.float32


def test_array_lookup_agrees_with_scalar_lookup(mapping):
    xs = np.arange(50, 350, 25, dtype=np.float32)
    ys = np.full_like(xs, 150.0)
    out_x, out_y = mapping.lookup(xs, ys)

    for i, (x, y) in enumerate(zip(xs, ys)):
        scalar = mapping.lookup(float(x), float(y))
        if scalar is None:
            assert np.isnan(out_x[i])
        else:
            assert out_x[i] == pytest.approx(scalar[0])
            assert out_y[i] == pytest.approx(scalar[1])


def test_array_lookup_agrees_with_flow(mapping):
    flow = mapping.flow(absolute=True)
    ys, xs = np.mgrid[10:290:17, 10:410:23]
    xs = xs.ravel().astype(np.float32)
    ys = ys.ravel().astype(np.float32)

    out_x, out_y = mapping.lookup(xs, ys)
    expected = flow[ys.astype(int), xs.astype(int)]
    np.testing.assert_allclose(out_x, expected[:, 0], rtol=1e-6)
    np.testing.assert_allclose(out_y, expected[:, 1], rtol=1e-6)


def test_mismatched_lookup_shapes_are_rejected(mapping):
    with pytest.raises(ValueError, match="same shape"):
        mapping.lookup(np.zeros(3, np.float32), np.zeros(4, np.float32))


def test_morph_returns_rgba_at_working_resolution(mapping):
    warped = mapping.morph(0.5)
    width, height = mapping.working_dimensions
    assert warped.shape == (height, width, 4)
    assert warped.dtype == np.uint8


def test_morph_output_is_fully_opaque(mapping):
    # The buffer is pre-filled opaque, so alpha says nothing about whether a pixel was
    # written. Pinned because the binding used to document alpha as the unfilled mask.
    assert np.all(mapping.morph(0.5)[..., 3] == 255)


def test_morph_leaves_unfilled_pixels_black(mapping):
    warped = mapping.morph(0.5)
    unfilled = (warped[..., :3] == 0).all(axis=2)
    # Some of the image is unmapped, but most of it is not.
    assert 0.0 < unfilled.mean() < 0.2


def test_morph_leaves_unfilled_pixels_where_the_mapping_is_missing(mapping):
    # At t=0 every mapped sample writes to its own position, so a black pixel means
    # exactly "nothing was mapped here" - which makes this the one interpolation value
    # where the unfilled mask can be checked against the flow field directly.
    unfilled = (mapping.morph(0.0)[..., :3] == 0).all(axis=2)

    height, width = unfilled.shape
    scale = mapping.working_scale
    ys, xs = np.mgrid[0:height, 0:width]
    out_x, _ = mapping.lookup(
        (xs / scale).ravel().astype(np.float32), (ys / scale).ravel().astype(np.float32)
    )
    unmapped = np.isnan(out_x).reshape(height, width)

    # Not identical: the scatter rounds to whole pixels, so a mapped sample can miss the
    # pixel it started on. Agreeing on the great majority is what rules out the old bug,
    # where every unmapped sample landed on (0, 0) instead of staying put.
    assert (unfilled == unmapped).mean() > 0.95


def test_morph_at_zero_leaves_the_photo_alone(mapping):
    start = mapping.morph(0.0)
    halfway = mapping.morph(1.0)
    assert not np.array_equal(start, halfway)


def test_morph_detail_must_be_at_least_one(mapping):
    with pytest.raises(ValueError, match="detail"):
        mapping.morph(0.5, detail=0)


def test_morph_with_more_detail_fills_more_pixels(mapping):
    def unfilled(warped):
        return (warped[..., :3] == 0).all(axis=2).sum()

    # Supersampling fills gaps the warp stretches open, which are left black.
    assert unfilled(mapping.morph(1.0, detail=2)) < unfilled(mapping.morph(1.0, detail=1))
