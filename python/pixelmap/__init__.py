"""Dense image correspondence: work out where every pixel of one photo went in another.

Python bindings for the `pixelmap <https://crates.io/crates/pixelmap>`_ Rust crate, the
reference implementation of the PIXELMAP framework. Every cell of an affine
correspondence grid acts as an autonomous agent holding its own local affine transform;
agents refine their transform against the image data and propagate what they find to
their neighbours, and a forward/backward consistency check culls the ones that disagree.

    >>> import numpy as np, pixelmap
    >>> mapping = pixelmap.correspond(photo_a, photo_b, quality="low")  # doctest: +SKIP
    >>> flow = mapping.flow()          # (H, W, 2) float32, NaN where unmapped
    >>> mapping.coverage               # doctest: +SKIP
    0.71

Regions the algorithm could not map — occlusions, featureless sky, anything the
consistency check rejected — come back as ``NaN`` rather than as a plausible-looking
coordinate. Coverage well below 1.0 is normal and not a failure.

Runs are deterministic: the same photos, quality and seed give the same mapping, run to
run and machine to machine.
"""

from __future__ import annotations

import enum
from typing import Callable, Optional, Union

import numpy as np

from . import _pixelmap
from ._errors import (
    DecodeError,
    PhotoTooSmallError,
    PixelmapError,
    SizeMismatchError,
)

__all__ = [
    "Correspondence",
    "Quality",
    "correspond",
    "MIN_DIMENSION",
    "PixelmapError",
    "SizeMismatchError",
    "PhotoTooSmallError",
    "DecodeError",
    "__version__",
]

try:  # pragma: no cover - trivial, and absent only in a broken install
    from importlib.metadata import version as _version

    # The distribution is `pixelmap-python`; only the import name is `pixelmap`. PyPI
    # refuses `pixelmap` as too similar to the unrelated `pixel-map` project.
    __version__ = _version("pixelmap-python")
except Exception:  # pragma: no cover
    __version__ = "unknown"

#: The smallest photo the pipeline can work with, on each side.
MIN_DIMENSION: int = _pixelmap.MIN_DIMENSION


class Quality(str, enum.Enum):
    """How much work to put into a photo pair.

    Each preset is a coarse-to-fine schedule that both runs more iterations and finishes
    at a higher working resolution, so cost grows faster than the names suggest: roughly
    400, 800 and 1600 pixels of working width respectively.
    """

    LOW = "low"
    """Fast, but may be less accurate. The default."""

    MEDIUM = "medium"
    """Slower, but more accurate."""

    HIGH = "high"
    """Slowest, but likely the best result."""

    def __str__(self) -> str:
        return self.value


QualityArg = Union[Quality, str]


def _quality_name(quality: QualityArg) -> str:
    if isinstance(quality, Quality):
        return quality.value
    if isinstance(quality, str):
        return quality
    raise TypeError(
        f"quality must be a pixelmap.Quality or one of 'low', 'medium', 'high', "
        f"got {type(quality).__name__}"
    )


def _as_image(image, name: str) -> np.ndarray:
    """Normalises anything array-like into a contiguous ``(H, W, 3|4)`` uint8 array.

    Accepts NumPy arrays, Pillow images, and anything else ``np.asarray`` understands.
    Grayscale input — ``(H, W)`` or ``(H, W, 1)`` — is broadcast to three channels, since
    the algorithm compares colour but nothing is lost by feeding it grey.
    """
    array = np.asarray(image)

    if array.dtype != np.uint8:
        raise ValueError(
            f"{name} must be a uint8 array, got dtype {array.dtype}. "
            f"Scale floating-point images to 0-255 and cast, e.g. "
            f"(img * 255).astype('uint8')."
        )

    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), "
            f"got {array.shape}"
        )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] not in (3, 4):
        raise ValueError(
            f"{name} must have 1, 3 or 4 channels, got {array.shape[2]} "
            f"(shape {array.shape})"
        )

    return np.ascontiguousarray(array)


class Correspondence:
    """A finished mapping between two photos, in both directions.

    Returned by :func:`correspond`; not constructed directly.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: _pixelmap.Correspondence) -> None:
        if not isinstance(inner, _pixelmap.Correspondence):
            raise TypeError(
                "Correspondence is returned by pixelmap.correspond(), not constructed "
                "directly"
            )
        self._inner = inner

    # -- geometry -------------------------------------------------------------

    def flow(self, *, backward: bool = False, absolute: bool = False) -> np.ndarray:
        """The dense correspondence field, as ``(H, W, 2)`` float32.

        ``flow[y, x]`` is ``(dx, dy)``: how far the pixel at ``(x, y)`` moved, in the
        source photos' own pixel coordinates. Unmapped pixels are ``NaN`` in both
        channels, so ``np.isnan(flow[..., 0])`` is the mask of what the algorithm could
        not place.

        Args:
            backward: Map the second photo back into the first instead.
            absolute: Return destination coordinates rather than displacements — what
                ``cv2.remap`` wants. Equivalent to adding the pixel grid to the default.

        Returns:
            An ``(H, W, 2)`` float32 array, ``H`` and ``W`` being the source dimensions.
        """
        return self._inner.flow(backward, absolute)

    def lookup(self, x, y, *, backward: bool = False):
        """Where the pixel at ``(x, y)`` ended up.

        Scalars in, ``(x, y)`` tuple or ``None`` out. Arrays in, a pair of float32
        arrays out with the input's shape and ``NaN`` where nothing was mapped.

        For a whole image use :meth:`flow`, which does the same work in one pass.
        """
        if np.isscalar(x) and np.isscalar(y):
            return self._inner.lookup_one(float(x), float(y), backward)

        xs = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        ys = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        if xs.shape != ys.shape:
            raise ValueError(
                f"x and y must have the same shape, got {xs.shape} and {ys.shape}"
            )
        out_x, out_y = self._inner.lookup_many(xs.ravel(), ys.ravel(), backward)
        return out_x.reshape(xs.shape), out_y.reshape(ys.shape)

    def morph(self, t: float, *, detail: int = 1) -> np.ndarray:
        """The first photo warped ``t`` of the way towards the second.

        ``t=0.0`` is the first photo unchanged and ``t=1.0`` is it fully warped onto the
        second.

        The warp scatters source pixels to their destinations, so pixels nothing lands on
        are left fully transparent — alpha is ``255`` or ``0`` and never in between, which
        makes ``warped[..., 3] == 0`` the mask of what the warp did not fill. Raising
        ``detail`` fills most of those in.

        The result is at the solver's *working* resolution, not the source resolution —
        see :attr:`working_dimensions` — because the warp is applied to the scaled photo
        the mapping was computed over.

        Args:
            t: How far to interpolate, clamped to ``0.0``-``1.0``.
            detail: Supersampling factor. Above 1 fills in gaps left where the warp
                stretches the image, at a quadratic cost.

        Returns:
            An ``(h, w, 4)`` uint8 RGBA array.
        """
        if detail < 1:
            raise ValueError(f"detail must be at least 1, got {detail}")
        return self._inner.morph(float(t), detail)

    # -- properties -----------------------------------------------------------

    @property
    def coverage(self) -> float:
        """The fraction of the first photo that got a usable mapping, ``0.0``-``1.0``.

        Well below 1.0 is normal: occlusions, featureless regions and anything the
        forward/backward consistency check rejected are all excluded. A very low value
        means the photos had little in common — or are related by something an affine
        grid cannot express.
        """
        return self._inner.coverage

    @property
    def comparisons(self) -> int:
        """How many region comparisons the solver performed."""
        return self._inner.comparisons

    @property
    def source_dimensions(self) -> tuple[int, int]:
        """``(width, height)`` of the photos this mapping was computed from."""
        return self._inner.source_dimensions

    @property
    def working_dimensions(self) -> tuple[int, int]:
        """``(width, height)`` the solver actually worked at."""
        return self._inner.working_dimensions

    @property
    def working_scale(self) -> float:
        """Source pixels to working-resolution pixels.

        Below 1.0 when the inputs were larger than the quality preset's working width.
        :meth:`flow` and :meth:`lookup` already account for it; this is here for callers
        who want to reason about the solver's effective resolution.
        """
        return self._inner.working_scale

    # -- serialization --------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Both mappings as bytes, to be read back by :meth:`from_bytes`.

        The photos are not included; :meth:`from_bytes` takes them again.
        """
        return self._inner.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes, photo1, photo2) -> Correspondence:
        """Rebuilds a mapping written by :meth:`to_bytes`.

        Args:
            data: Bytes from :meth:`to_bytes`.
            photo1: The first photo the mapping was computed from.
            photo2: The second photo.

        Raises:
            DecodeError: If the data is not a mapping this build can read, or the photos
                are not the size the mapping was computed at.
        """
        return cls(
            _pixelmap.Correspondence.from_bytes(
                bytes(data), _as_image(photo1, "photo1"), _as_image(photo2, "photo2")
            )
        )

    def __repr__(self) -> str:
        width, height = self.source_dimensions
        return f"<pixelmap.Correspondence {width}x{height}, coverage {self.coverage:.1%}>"


def correspond(
    photo1,
    photo2,
    *,
    quality: QualityArg = Quality.LOW,
    seed: Optional[int] = None,
    max_round_trip_error: float = 2.0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Correspondence:
    """Maps two photos onto each other.

    Args:
        photo1: The first photo, as a uint8 array of shape ``(H, W)``, ``(H, W, 1)``,
            ``(H, W, 3)`` or ``(H, W, 4)``. Pillow images work directly.
        photo2: The second photo. Must have the same dimensions as the first.
        quality: :class:`Quality` preset, or the string ``"low"``, ``"medium"`` or
            ``"high"``. Higher settings run more iterations at a higher working
            resolution.
        seed: Seeds the solver's queue shuffling. Two runs agreeing on inputs, quality
            and seed produce the same mapping. Defaults to the crate's own fixed seed,
            so runs are reproducible without asking.
        max_round_trip_error: How far a round trip through both mappings may land from
            where it started, in working-resolution pixels, before a cell is discarded.
            Lower keeps only what both directions agree on closely, at the cost of
            coverage; higher keeps more, including some that are wrong.
        progress: Called as ``progress(step, total)`` after the initial matching pass and
            after every schedule step. An exception raised here stops further callbacks
            and is re-raised once the run finishes; Ctrl-C is surfaced the same way.

    Returns:
        A :class:`Correspondence`.

    Raises:
        SizeMismatchError: The photos have different dimensions.
        PhotoTooSmallError: A photo is smaller than :data:`MIN_DIMENSION` on a side.
        ValueError: A photo is empty or not a uint8 array of an accepted shape, or
            ``quality`` does not name a preset.
    """
    if max_round_trip_error <= 0:
        raise ValueError(
            f"max_round_trip_error must be positive, got {max_round_trip_error}"
        )
    if seed is not None and not (0 <= seed < 2**64):
        raise ValueError(f"seed must fit in an unsigned 64-bit integer, got {seed}")

    return Correspondence(
        _pixelmap.correspond(
            _as_image(photo1, "photo1"),
            _as_image(photo2, "photo2"),
            _quality_name(quality),
            seed,
            float(max_round_trip_error),
            progress,
        )
    )
