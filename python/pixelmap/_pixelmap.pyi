"""Type stubs for the native module.

Not the package's public interface — see ``pixelmap/__init__.py`` for that. These exist so
that type checkers can follow the wrapper's calls into Rust.
"""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

MIN_DIMENSION: int

class Correspondence:
    @property
    def comparisons(self) -> int: ...
    @property
    def coverage(self) -> float: ...
    @property
    def source_dimensions(self) -> tuple[int, int]: ...
    @property
    def working_dimensions(self) -> tuple[int, int]: ...
    @property
    def working_scale(self) -> float: ...
    def flow(self, backward: bool = ..., absolute: bool = ...) -> NDArray[np.float32]: ...
    def lookup_one(
        self, x: float, y: float, backward: bool = ...
    ) -> tuple[float, float] | None: ...
    def lookup_many(
        self,
        xs: NDArray[np.float32],
        ys: NDArray[np.float32],
        backward: bool = ...,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
    def morph(self, t: float, detail: int = ...) -> NDArray[np.uint8]: ...
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(
        data: bytes, photo1: NDArray[np.uint8], photo2: NDArray[np.uint8]
    ) -> Correspondence: ...

def correspond(
    photo1: NDArray[np.uint8],
    photo2: NDArray[np.uint8],
    quality: str,
    seed: int | None,
    max_round_trip_error: float,
    progress: Callable[[int, int], Any] | None,
) -> Correspondence: ...
