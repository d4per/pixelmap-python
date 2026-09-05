"""The progress callback, and what happens when it misbehaves."""

import pixelmap
import pytest


def test_progress_is_reported_from_first_step_to_last(small_pair):
    steps = []
    pixelmap.correspond(
        *small_pair, progress=lambda step, total: steps.append((step, total))
    )

    assert steps, "the callback was never called"
    totals = {total for _, total in steps}
    assert len(totals) == 1, f"the reported total changed mid-run: {totals}"
    total = totals.pop()

    # The initial matching pass counts as a step, so the callbacks cover 1..total.
    assert [step for step, _ in steps] == list(range(1, total + 1))


def test_higher_quality_reports_more_steps(small_pair):
    def count(quality):
        steps = []
        pixelmap.correspond(
            *small_pair, quality=quality, progress=lambda s, t: steps.append(t)
        )
        return steps[0]

    assert count("medium") > count("low")


def test_an_exception_from_the_callback_reaches_the_caller(small_pair):
    calls = []

    def explode(step, total):
        calls.append(step)
        raise RuntimeError("stop right there")

    with pytest.raises(RuntimeError, match="stop right there"):
        pixelmap.correspond(*small_pair, progress=explode)

    # The run cannot be aborted mid-flight, but the callback is not called again.
    assert calls == [1]


def test_no_callback_is_the_default(small_pair):
    assert pixelmap.correspond(*small_pair).coverage > 0.5
