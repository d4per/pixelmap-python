//! Translating `pixelmap::Error` into Python exceptions.

use pixelmap::Error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Raises the Python exception that corresponds to `err`.
///
/// The classes live in `pixelmap._errors` rather than being created here: they inherit
/// from both `PixelmapError` and `ValueError`, so that `except ValueError` keeps working
/// for callers who do not know this library's hierarchy. Two bases is a plain class
/// statement in Python and a fight with `create_exception!` in Rust.
pub fn to_pyerr(py: Python<'_>, err: Error) -> PyErr {
    let message = err.to_string();
    match err {
        Error::SizeMismatch { first, second } => {
            build(py, "SizeMismatchError", &message, |kwargs| {
                kwargs.set_item("first", first)?;
                kwargs.set_item("second", second)
            })
        }
        Error::PhotoTooSmall {
            dimensions,
            minimum,
        } => build(py, "PhotoTooSmallError", &message, |kwargs| {
            kwargs.set_item("dimensions", dimensions)?;
            kwargs.set_item("minimum", minimum)
        }),
        Error::Decode(_) => build(py, "DecodeError", &message, |_| Ok(())),
        // `EmptyPhoto` and `BufferLength` say the array handed over was the wrong shape,
        // which is what `ValueError` already means; a dedicated class would earn nothing.
        _ => PyValueError::new_err(message),
    }
}

fn build(
    py: Python<'_>,
    class_name: &str,
    message: &str,
    fill: impl FnOnce(&Bound<'_, PyDict>) -> PyResult<()>,
) -> PyErr {
    let construct = || -> PyResult<PyErr> {
        let class = py.import("pixelmap._errors")?.getattr(class_name)?;
        let kwargs = PyDict::new(py);
        fill(&kwargs)?;
        Ok(PyErr::from_value(class.call((message,), Some(&kwargs))?))
    };
    // Failing to build the exception is itself an exception worth surfacing; it means the
    // Python half of the package is missing or out of step with this module.
    construct().unwrap_or_else(|failure| failure)
}
