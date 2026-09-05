//! Moving pixels between NumPy arrays and the crate's `Photo`.

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pixelmap::{Error, Photo};
use pyo3::prelude::*;

/// Builds a `Photo` from an `(H, W, 3)` or `(H, W, 4)` uint8 array.
///
/// The Python layer has already normalised dtype, rank and channel count, so the only
/// thing left to handle here is a stride pattern NumPy would not flatten for us.
pub fn photo_from_array(array: &PyReadonlyArray3<'_, u8>) -> Result<Photo, Error> {
    let view = array.as_array();
    let shape = view.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);

    // `as_slice` is None for a non-contiguous view. The Python layer calls
    // `ascontiguousarray`, so this is belt and braces rather than the expected path —
    // but copying is still better than panicking if that ever stops being true.
    let copied;
    let data: &[u8] = match view.as_slice() {
        Some(slice) => slice,
        None => {
            copied = view.iter().copied().collect::<Vec<u8>>();
            &copied
        }
    };

    if channels == 4 {
        Photo::from_rgba(width, height, data.to_vec())
    } else {
        Photo::from_rgb(width, height, data)
    }
}

/// Hands a photo's pixels back to Python as an `(H, W, 4)` uint8 array.
pub fn array_from_photo(py: Python<'_>, photo: Photo) -> Bound<'_, PyArray3<u8>> {
    let (width, height) = (photo.width(), photo.height());
    Array3::from_shape_vec((height, width, 4), photo.into_rgba())
        .expect("a Photo's buffer is always width * height * 4 bytes")
        .into_pyarray(py)
}
