# pixelmap-python

[![PyPI](https://img.shields.io/pypi/v/pixelmap-python.svg)](https://pypi.org/project/pixelmap-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/pixelmap-python.svg)](https://pypi.org/project/pixelmap-python/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Dense image correspondence: given two photographs of the same scene, work out where
each pixel of the first one went in the second.**

Python bindings for the [`pixelmap`](https://crates.io/crates/pixelmap) Rust crate, the
reference implementation of the PIXELMAP framework
([white paper](https://doi.org/10.36227/techrxiv.173749998.89779329/v1)). Every cell of an
*affine correspondence grid* acts as an autonomous agent holding its own local affine
transform; agents refine their transform against the image data and propagate what they
find to their neighbours, and a forward/backward consistency check culls the ones that
disagree. Repeating that coarse-to-fine yields a dense, geometrically consistent mapping.

Useful for optical flow, image registration and stitching, stereo matching, morphing, and
as the front half of a 3D reconstruction.

![PIXELMAP applied to two photos of a monkey statue](https://raw.githubusercontent.com/d4per/pixelmap/main/images/apa_3.png)

## Install

```console
pip install pixelmap-python
```

The package installs as `pixelmap-python` but imports as `pixelmap` — PyPI will not accept
`pixelmap` as a distribution name, because it collides with the unrelated `pixel-map`
project under PyPI's similarity rule.

Wheels are published for Linux (x86-64, aarch64, musl), macOS (Apple silicon and Intel)
and Windows (x86-64), for CPython 3.9 and newer. NumPy is the only runtime dependency; no
Rust toolchain is needed unless you build from source.

## Quick start

```python
import numpy as np
import pixelmap
from PIL import Image

a = np.asarray(Image.open("a.jpg").convert("RGB"))
b = np.asarray(Image.open("b.jpg").convert("RGB"))  # same dimensions as a

mapping = pixelmap.correspond(a, b, quality="low")

flow = mapping.flow()  # (H, W, 2) float32: how far each pixel moved
print(f"{mapping.coverage:.1%} of the image was mapped")

# Where did the pixel at (120, 84) end up?
print(mapping.lookup(120.0, 84.0))  # (114.2, 80.6), or None if unmapped
```

`flow[y, x]` is `(dx, dy)` in the source photos' own pixel coordinates. Regions the
algorithm could not map — occlusions, featureless sky, anything the consistency check
rejected — are `NaN` rather than a plausible-looking coordinate:

```python
unmapped = np.isnan(flow[..., 0])
```

Coverage well below 1.0 is normal and not a failure. A very low value means the two photos
had little in common, or are related by something an affine grid cannot express.

### Warping one photo onto the other

`flow(absolute=True)` returns destination coordinates instead of displacements, which is
what OpenCV's `remap` wants:

```python
import cv2

dst = mapping.flow(absolute=True)
warped = cv2.remap(a, dst[..., 0], dst[..., 1], cv2.INTER_LINEAR)
```

There is also a built-in morph, which interpolates the first photo `t` of the way towards
the second:

```python
halfway = mapping.morph(0.5, detail=2)  # (h, w, 4) uint8, at working resolution
```

### Watching a long run

```python
mapping = pixelmap.correspond(
    a,
    b,
    quality=pixelmap.Quality.HIGH,
    progress=lambda step, total: print(f"{step}/{total}"),
)
```

The GIL is released while the solver runs, so `correspond` can be called from a worker
thread without blocking the rest of your program.

## API

| | |
| --- | --- |
| `correspond(photo1, photo2, *, quality, seed, max_round_trip_error, progress)` | Run the pipeline. |
| `Correspondence.flow(*, backward=False, absolute=False)` | The dense field as `(H, W, 2)` float32. |
| `Correspondence.lookup(x, y, *, backward=False)` | One point, or NumPy arrays of them. |
| `Correspondence.morph(t, *, detail=1)` | The first photo warped towards the second. |
| `Correspondence.coverage` | Fraction of the image that got a mapping. |
| `Correspondence.comparisons` | Region comparisons performed. |
| `Correspondence.source_dimensions` / `.working_dimensions` / `.working_scale` | Geometry. |
| `Correspondence.to_bytes()` / `.from_bytes(data, photo1, photo2)` | Save and reload a mapping. |
| `Quality.LOW` / `.MEDIUM` / `.HIGH` | Presets, or the equivalent strings. |

**Input.** Photos are `uint8` arrays of shape `(H, W)`, `(H, W, 1)`, `(H, W, 3)` or
`(H, W, 4)` — Pillow images work directly. Both must have the same dimensions and be at
least 32 pixels on each side. Violations raise `SizeMismatchError`, `PhotoTooSmallError`
or `ValueError`, all of which are `ValueError` subclasses.

**Quality.** `LOW`, `MEDIUM` and `HIGH` run 4, 10 and 13 refinement steps and finish at a
working width of 400, 800 and 1600 pixels respectively, so cost grows faster than the step
count suggests. `flow` and `lookup` answer in your photos' coordinates regardless; use
`working_scale` if you need to reason about the solver's effective resolution.

**Determinism.** The same photos, quality and seed give the same mapping — run to run,
thread to thread, and machine to machine. Pass `seed=` to vary it.

## Building from source

Requires Rust 1.83 or newer.

```console
git clone https://github.com/d4per/pixelmap-python
cd pixelmap-python
pip install maturin
maturin develop --release
pytest
```

`maturin develop` without `--release` builds an unoptimised solver that is many times
slower; use it only for iterating on the bindings themselves.

## See also

- [Interactive demo](https://pixelmap.dogduck.com/) — upload your own images.
- [`pixelmap` on crates.io](https://crates.io/crates/pixelmap) and its
  [API docs](https://docs.rs/pixelmap) — the Rust library this wraps.
- [The PIXELMAP repository](https://github.com/d4per/pixelmap) — the algorithm, a command
  line tool, a viewer, and 3D reconstruction.
- [White paper](https://doi.org/10.36227/techrxiv.173749998.89779329/v1).

## License

MIT — see [LICENSE](LICENSE).
