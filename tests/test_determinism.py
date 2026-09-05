"""Reproducibility is a contract, not an accident."""

import numpy as np
import pixelmap
from conftest import shifted_pair


def test_the_same_seed_reproduces_the_same_mapping():
    first = pixelmap.correspond(*shifted_pair(200, 150, 5, 2), seed=4242).flow()
    second = pixelmap.correspond(*shifted_pair(200, 150, 5, 2), seed=4242).flow()
    np.testing.assert_array_equal(first, second)


def test_the_default_seed_is_also_reproducible():
    first = pixelmap.correspond(*shifted_pair(200, 150, 5, 2)).flow()
    second = pixelmap.correspond(*shifted_pair(200, 150, 5, 2)).flow()
    np.testing.assert_array_equal(first, second)


def test_different_seeds_explore_differently():
    """...so the test above is not passing because the solver ignores the seed."""
    first = pixelmap.correspond(*shifted_pair(200, 150, 5, 2), seed=1).flow()
    second = pixelmap.correspond(*shifted_pair(200, 150, 5, 2), seed=999_999).flow()
    assert not np.array_equal(np.nan_to_num(first), np.nan_to_num(second))


def test_a_mapping_can_be_computed_on_a_worker_thread(small_pair):
    from concurrent.futures import ThreadPoolExecutor

    # Also checks the GIL is actually released: four runs on four threads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [
            pool.submit(pixelmap.correspond, *small_pair, seed=77).result()
            for _ in range(4)
        ]

    alone = pixelmap.correspond(*small_pair, seed=77).flow()
    for concurrent in results:
        np.testing.assert_array_equal(concurrent.flow(), alone)


def test_lower_round_trip_tolerance_keeps_less(pair):
    strict = pixelmap.correspond(*pair, max_round_trip_error=0.1)
    loose = pixelmap.correspond(*pair, max_round_trip_error=5.0)
    assert strict.coverage < loose.coverage
