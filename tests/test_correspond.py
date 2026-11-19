"""The headline claim: given a known warp, the pipeline recovers it."""

import numpy as np
import pixelmap
import pytest
from conftest import shifted_pair


def test_recovers_a_known_translation(pair):
    a, b = pair
    mapping = pixelmap.correspond(a, b, quality="low")

    flow = mapping.flow()
    assert flow.shape == (300, 420, 2)
    assert flow.dtype == np.float32

    mapped = ~np.isnan(flow[..., 0])
    coverage = mapped.mean()
    assert coverage > 0.8, f"only {coverage:.1%} of the image was mapped"

    # The pair was built by shifting the window right and down, so the content moved the
    # other way: the forward flow is (-dx, -dy) in the source photos' own coordinates.
    assert np.nanmean(flow[..., 0]) == pytest.approx(-6.0, abs=0.5)
    assert np.nanmean(flow[..., 1]) == pytest.approx(-3.0, abs=0.5)


def test_unmapped_pixels_are_nan_in_both_channels(pair):
    flow = pixelmap.correspond(*pair).flow()
    assert np.array_equal(np.isnan(flow[..., 0]), np.isnan(flow[..., 1]))


def test_absolute_flow_is_the_displacement_plus_the_pixel_grid(pair):
    mapping = pixelmap.correspond(*pair)
    relative = mapping.flow()
    absolute = mapping.flow(absolute=True)

    height, width = relative.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    np.testing.assert_allclose(absolute[..., 0], relative[..., 0] + xs, rtol=1e-5)
    np.testing.assert_allclose(absolute[..., 1], relative[..., 1] + ys, rtol=1e-5)


def test_backward_flow_undoes_the_forward_one(pair):
    mapping = pixelmap.correspond(*pair)
    backward = mapping.flow(backward=True)

    # The reverse of a (-6, -3) translation.
    assert np.nanmean(backward[..., 0]) == pytest.approx(6.0, abs=0.5)
    assert np.nanmean(backward[..., 1]) == pytest.approx(3.0, abs=0.5)


def test_round_trip_through_both_directions_lands_where_it_started(pair):
    mapping = pixelmap.correspond(*pair)

    ys, xs = np.mgrid[40:260:20, 40:380:20]
    xs = xs.ravel().astype(np.float32)
    ys = ys.ravel().astype(np.float32)

    mid_x, mid_y = mapping.lookup(xs, ys)
    back_x, back_y = mapping.lookup(mid_x, mid_y, backward=True)

    ok = ~np.isnan(back_x)
    assert ok.sum() > 0.5 * ok.size
    # The consistency check culls anything further out than max_round_trip_error, scaled
    # from working resolution back into source pixels.
    tolerance = 2.0 / mapping.working_scale + 1.0
    assert np.all(np.abs(back_x[ok] - xs[ok]) < tolerance)
    assert np.all(np.abs(back_y[ok] - ys[ok]) < tolerance)


def test_metadata_describes_the_run(pair):
    mapping = pixelmap.correspond(*pair, quality="low")

    assert mapping.source_dimensions == (420, 300)
    assert 0.0 < mapping.coverage <= 1.0
    assert mapping.comparisons > 0
    # Quality.LOW finishes at a working width of 400.
    assert mapping.working_dimensions[0] == 400
    assert mapping.working_scale == pytest.approx(400 / 420)
    assert "420x300" in repr(mapping)


@pytest.mark.parametrize("quality", ["low", pixelmap.Quality.LOW])
def test_quality_accepts_strings_and_enum_members(small_pair, quality):
    assert pixelmap.correspond(*small_pair, quality=quality).coverage > 0.5


def test_higher_quality_works_at_a_higher_resolution():
    pair = shifted_pair(1000, 160, 5, 2)
    low = pixelmap.correspond(*pair, quality="low")
    medium = pixelmap.correspond(*pair, quality="medium")
    assert low.working_dimensions[0] == 400
    assert medium.working_dimensions[0] == 800
    assert medium.comparisons > low.comparisons


def test_above_low_no_step_works_wider_than_the_photos(small_pair):
    for quality in (pixelmap.Quality.MEDIUM, pixelmap.Quality.HIGH, pixelmap.Quality.ULTRA):
        mapping = pixelmap.correspond(*small_pair, quality=quality)
        assert mapping.working_dimensions[0] == small_pair[0].shape[1]
        assert mapping.coverage > 0.5


def test_ultra_is_high_on_photos_up_to_1600_pixels_wide(small_pair):
    high = pixelmap.correspond(*small_pair, quality="high")
    ultra = pixelmap.correspond(*small_pair, quality=pixelmap.Quality.ULTRA)
    assert np.array_equal(high.flow(), ultra.flow(), equal_nan=True)


def test_fill_mode_defaults_to_the_preset_and_can_be_overridden(pair):
    a, b = pair
    default = pixelmap.correspond(a, b, quality="medium")
    plain = pixelmap.correspond(a, b, quality="medium", fill_mode="plain")
    off = pixelmap.correspond(a, b, quality="medium", fill_mode="off")
    assert np.array_equal(default.flow(), plain.flow(), equal_nan=True)
    assert plain.coverage >= off.coverage
    with pytest.raises(ValueError):
        pixelmap.correspond(a, b, fill_mode="sometimes")
