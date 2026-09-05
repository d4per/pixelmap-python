"""The exceptions this package raises.

Each one inherits from :class:`PixelmapError` and from a built-in exception, so callers
who do not know this hierarchy still catch what they expect: every one of these means
"the arguments were wrong", which is what ``ValueError`` already says.

The classes live here rather than in the native module because the Rust side constructs
them by importing this module, and because two base classes is a plain class statement in
Python.
"""

__all__ = [
    "PixelmapError",
    "SizeMismatchError",
    "PhotoTooSmallError",
    "DecodeError",
]


class PixelmapError(Exception):
    """Base class for every error raised by :mod:`pixelmap`."""


class SizeMismatchError(PixelmapError, ValueError):
    """The two photos had different dimensions.

    The mapping is defined over a shared pixel grid, so the inputs have to agree on one.
    Rescale or crop before calling.

    Attributes:
        first: ``(width, height)`` of the first photo.
        second: ``(width, height)`` of the second photo.
    """

    def __init__(self, message, first=None, second=None):
        super().__init__(message)
        self.first = first
        self.second = second


class PhotoTooSmallError(PixelmapError, ValueError):
    """A photo was too small for the feature detector's sampling disc.

    Attributes:
        dimensions: ``(width, height)`` of the offending photo.
        minimum: The smallest width and height the pipeline can work with.
    """

    def __init__(self, message, dimensions=None, minimum=None):
        super().__init__(message)
        self.dimensions = dimensions
        self.minimum = minimum


class DecodeError(PixelmapError, ValueError):
    """Serialized mapping data could not be read back."""
