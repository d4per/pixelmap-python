"""A round trip through to_bytes/from_bytes must preserve every mapped coordinate."""

import numpy as np
import pixelmap
import pytest


def test_round_trips_exactly(pair):
    a, b = pair
    original = pixelmap.correspond(a, b)
    restored = pixelmap.Correspondence.from_bytes(original.to_bytes(), a, b)

    np.testing.assert_array_equal(original.flow(), restored.flow())
    np.testing.assert_array_equal(
        original.flow(backward=True), restored.flow(backward=True)
    )
    assert restored.source_dimensions == original.source_dimensions
    assert restored.working_dimensions == original.working_dimensions
    assert restored.comparisons == original.comparisons
    assert restored.coverage == pytest.approx(original.coverage)


def test_restored_mapping_can_still_morph(small_pair):
    a, b = small_pair
    original = pixelmap.correspond(a, b)
    restored = pixelmap.Correspondence.from_bytes(original.to_bytes(), a, b)
    np.testing.assert_array_equal(original.morph(0.5), restored.morph(0.5))


@pytest.mark.parametrize(
    ("mangle", "message"),
    [
        (lambda data: b"XXXX" + data[4:], "not a pixelmap mapping"),
        (lambda data: data[:4] + b"\x63\x00" + data[6:], "format version"),
        (lambda data: data[:20], "header"),
        (lambda data: data[:60], "truncated"),
    ],
)
def test_corrupt_data_is_rejected(small_pair, mangle, message):
    a, b = small_pair
    data = pixelmap.correspond(a, b).to_bytes()
    with pytest.raises(pixelmap.DecodeError, match=message):
        pixelmap.Correspondence.from_bytes(mangle(data), a, b)


def test_wrong_photos_are_rejected(small_pair, pair):
    data = pixelmap.correspond(*small_pair).to_bytes()
    with pytest.raises(pixelmap.DecodeError, match="200x150"):
        pixelmap.Correspondence.from_bytes(data, *pair)


def test_decode_error_is_also_a_value_error(small_pair):
    a, b = small_pair
    with pytest.raises(ValueError):
        pixelmap.Correspondence.from_bytes(b"nonsense", a, b)
