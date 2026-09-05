"""Synthetic photo pairs with a known warp between them.

Ported from the crate's own integration tests (`rust/pixelmap/tests/api.rs`) so that both
test suites judge the binding against the same ground truth: two windows onto one texture,
offset by a known amount, whose correct forward flow is exactly ``(-dx, -dy)``.
"""

import numpy as np
import pytest


def texture(width: int, height: int) -> np.ndarray:
    """A texture with enough local structure for the feature matcher to lock onto.

    A smooth interference pattern so neighbouring pixels differ, plus noise so distant
    regions do not look alike.
    """
    rng = np.random.default_rng(0x1234_5678)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    wave = 127.0 + 100.0 * np.sin(x / 9.0) * np.cos(y / 11.0)
    wave = wave + rng.integers(0, 70, size=(height, width)) - 35.0
    v = np.clip(wave, 0, 255).astype(np.uint8)

    image = np.empty((height, width, 4), dtype=np.uint8)
    image[..., 0] = v
    image[..., 1] = np.clip(v.astype(np.float32) * 0.7 + 40.0, 0, 255).astype(np.uint8)
    image[..., 2] = 255 - v
    image[..., 3] = 255
    return image


def shifted_pair(width: int, height: int, dx: int, dy: int):
    """Two windows onto the same texture, offset by ``(dx, dy)``.

    ``b`` samples the base image ``(dx, dy)`` further along than ``a`` does, so whatever
    ``a`` shows at ``(x, y)`` sits at ``(x - dx, y - dy)`` in ``b``: the forward mapping
    is a translation of ``(-dx, -dy)``.
    """
    base = texture(width + dx, height + dy)
    a = base[0:height, 0:width].copy()
    b = base[dy : dy + height, dx : dx + width].copy()
    return a, b


@pytest.fixture(scope="session")
def pair():
    """A 420x300 pair shifted by (6, 3), large enough for meaningful coverage."""
    return shifted_pair(420, 300, 6, 3)


@pytest.fixture(scope="session")
def small_pair():
    """A 200x150 pair shifted by (5, 2), for tests that only need *a* mapping."""
    return shifted_pair(200, 150, 5, 2)
