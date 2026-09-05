//! Native half of the `pixelmap` Python package.
//!
//! Everything user-facing — argument normalisation, docstrings, the exception classes —
//! lives in the Python layer next door. This module does the FFI and nothing else.

mod convert;
mod errors;

use std::str::FromStr;
use std::sync::Arc;

use numpy::ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pixelmap::{DensePhotoMap, Quality, MIN_DIMENSION};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::convert::{array_from_photo, photo_from_array};
use crate::errors::to_pyerr;

/// Magic number and version for the container `to_bytes` writes. Distinct from the
/// crate's own `PXMP`, which this format wraps two of.
const MAGIC: &[u8; 4] = b"PXPY";
const FORMAT_VERSION: u16 = 1;

/// A finished mapping between two photos, in both directions.
///
/// The fields mirror `pixelmap::Correspondence`. Holding the parts ourselves rather than
/// the crate's struct is what makes `from_bytes` possible: the crate has no constructor
/// from a pair of maps, so a `Correspondence` that came off disc could not be rebuilt.
/// The cost is the two coordinate-scaling methods below, copied from `Correspondence`.
#[pyclass(module = "pixelmap._pixelmap", frozen)]
pub struct Correspondence {
    forward: DensePhotoMap,
    backward: DensePhotoMap,
    comparisons: usize,
    source_size: (usize, usize),
}

impl Correspondence {
    /// The map for one direction, together with the factor between source pixels and the
    /// working resolution it is expressed in.
    fn direction(&self, backward: bool) -> (&DensePhotoMap, f32) {
        let map = if backward {
            &self.backward
        } else {
            &self.forward
        };
        (map, map.dimensions().0 as f32 / self.source_size.0 as f32)
    }

    /// Where the source pixel at `(x, y)` ended up, in source coordinates.
    fn map_point(&self, x: f32, y: f32, backward: bool) -> Option<(f32, f32)> {
        let (map, scale) = self.direction(backward);
        let (mx, my) = map.lookup(x * scale, y * scale)?;
        Some((mx / scale, my / scale))
    }

    /// The working width both maps finished at, needed to rebuild the scaled photos when
    /// deserializing.
    fn working_width(&self) -> usize {
        self.forward.dimensions().0
    }
}

#[pymethods]
impl Correspondence {
    /// How many region comparisons the solver performed.
    #[getter]
    fn comparisons(&self) -> usize {
        self.comparisons
    }

    /// The fraction of the first photo that ended up with a usable mapping.
    #[getter]
    fn coverage(&self) -> f32 {
        self.forward.calculate_used_area()
    }

    /// The `(width, height)` of the photos this mapping was computed from.
    #[getter]
    fn source_dimensions(&self) -> (usize, usize) {
        self.source_size
    }

    /// The `(width, height)` the solver actually worked at.
    #[getter]
    fn working_dimensions(&self) -> (usize, usize) {
        self.forward.dimensions()
    }

    /// Source pixels to working-resolution pixels.
    #[getter]
    fn working_scale(&self) -> f32 {
        self.direction(false).1
    }

    /// The dense field as `(H, W, 2)` float32, `NaN` where nothing was mapped.
    #[pyo3(signature = (backward = false, absolute = false))]
    fn flow<'py>(
        &self,
        py: Python<'py>,
        backward: bool,
        absolute: bool,
    ) -> Bound<'py, PyArray3<f32>> {
        let (width, height) = self.source_size;
        // Millions of lookups with no Python involved: there is no reason for the
        // caller's other threads to wait on us.
        let data = py.detach(|| {
            let mut data = Vec::with_capacity(width * height * 2);
            for y in 0..height {
                for x in 0..width {
                    let (sx, sy) = (x as f32, y as f32);
                    match self.map_point(sx, sy, backward) {
                        Some((mx, my)) if absolute => data.extend_from_slice(&[mx, my]),
                        Some((mx, my)) => data.extend_from_slice(&[mx - sx, my - sy]),
                        None => data.extend_from_slice(&[f32::NAN, f32::NAN]),
                    }
                }
            }
            data
        });
        Array3::from_shape_vec((height, width, 2), data)
            .expect("the loop pushes exactly height * width * 2 values")
            .into_pyarray(py)
    }

    /// One point, or `None` where nothing was mapped.
    #[pyo3(signature = (x, y, backward = false))]
    fn lookup_one(&self, x: f32, y: f32, backward: bool) -> Option<(f32, f32)> {
        self.map_point(x, y, backward)
    }

    /// Many points at once, `NaN` where nothing was mapped.
    #[pyo3(signature = (xs, ys, backward = false))]
    fn lookup_many<'py>(
        &self,
        py: Python<'py>,
        xs: PyReadonlyArray1<'py, f32>,
        ys: PyReadonlyArray1<'py, f32>,
        backward: bool,
    ) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
        let (xs, ys) = (xs.as_array(), ys.as_array());
        let mut out_x = Vec::with_capacity(xs.len());
        let mut out_y = Vec::with_capacity(xs.len());
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            match self.map_point(x, y, backward) {
                Some((mx, my)) => {
                    out_x.push(mx);
                    out_y.push(my);
                }
                None => {
                    out_x.push(f32::NAN);
                    out_y.push(f32::NAN);
                }
            }
        }
        (
            Array1::from_vec(out_x).into_pyarray(py),
            Array1::from_vec(out_y).into_pyarray(py),
        )
    }

    /// The first photo warped `t` of the way towards the second, at working resolution.
    #[pyo3(signature = (t, detail = 1))]
    fn morph<'py>(&self, py: Python<'py>, t: f32, detail: usize) -> Bound<'py, PyArray3<u8>> {
        let photo = py.detach(|| self.forward.interpolate_photo(t, detail));
        array_from_photo(py, photo)
    }

    /// Both maps and the metadata needed to rebuild this object.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let forward = self.forward.serialize();
        let backward = self.backward.serialize();

        let mut out = Vec::with_capacity(48 + forward.len() + backward.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        for field in [
            self.source_size.0,
            self.source_size.1,
            self.working_width(),
            self.comparisons,
            forward.len(),
        ] {
            out.extend_from_slice(&(field as u64).to_le_bytes());
        }
        out.extend_from_slice(&forward);
        out.extend_from_slice(&backward);
        PyBytes::new(py, &out)
    }

    /// Rebuilds a mapping written by `to_bytes`, pairing it with its photos again.
    #[staticmethod]
    fn from_bytes(
        py: Python<'_>,
        data: &[u8],
        photo1: PyReadonlyArray3<'_, u8>,
        photo2: PyReadonlyArray3<'_, u8>,
    ) -> PyResult<Correspondence> {
        let read_u64 = |offset: usize| -> PyResult<usize> {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|slice| slice.try_into().ok())
                .ok_or_else(|| decode_error(py, "mapping data ends mid-header"))?;
            Ok(u64::from_le_bytes(bytes) as usize)
        };

        if data.len() < 6 || &data[..4] != MAGIC {
            return Err(decode_error(py, "not a pixelmap mapping"));
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != FORMAT_VERSION {
            return Err(decode_error(
                py,
                &format!("mapping format version {version}, but this build reads version {FORMAT_VERSION}"),
            ));
        }

        let source_size = (read_u64(6)?, read_u64(14)?);
        let working_width = read_u64(22)?;
        let comparisons = read_u64(30)?;
        let forward_len = read_u64(38)?;
        let body = &data[46..];
        if forward_len > body.len() {
            return Err(decode_error(py, "mapping data is truncated"));
        }
        let (forward_bytes, backward_bytes) = body.split_at(forward_len);

        let photo1 = photo_from_array(&photo1).map_err(|err| to_pyerr(py, err))?;
        let photo2 = photo_from_array(&photo2).map_err(|err| to_pyerr(py, err))?;
        if (photo1.width(), photo1.height()) != source_size {
            return Err(decode_error(
                py,
                &format!(
                    "mapping was computed from {}x{} photos, but got {}x{}",
                    source_size.0,
                    source_size.1,
                    photo1.width(),
                    photo1.height()
                ),
            ));
        }

        // The maps hold the photos at the working resolution the solver finished at, and
        // `scaled_to_width` is exactly how the pipeline produced them, so rescaling here
        // reconstructs the same buffers rather than something merely similar.
        if working_width == 0 {
            return Err(decode_error(py, "mapping declares a zero working width"));
        }
        let scaled1 = Arc::new(photo1.scaled_to_width(working_width));
        let scaled2 = Arc::new(photo2.scaled_to_width(working_width));

        let forward = DensePhotoMap::deserialize(forward_bytes, scaled1.clone(), scaled2.clone())
            .map_err(|err| to_pyerr(py, err))?;
        let backward = DensePhotoMap::deserialize(backward_bytes, scaled2, scaled1)
            .map_err(|err| to_pyerr(py, err))?;

        Ok(Correspondence {
            forward,
            backward,
            comparisons,
            source_size,
        })
    }
}

/// A `DecodeError` for a problem with this module's own container format, as opposed to
/// the crate's, which arrives as a `pixelmap::Error` already.
fn decode_error(py: Python<'_>, message: &str) -> PyErr {
    match py
        .import("pixelmap._errors")
        .and_then(|module| module.getattr("DecodeError"))
        .and_then(|class| class.call1((message,)))
    {
        Ok(instance) => PyErr::from_value(instance),
        Err(failure) => failure,
    }
}

/// Maps two photos onto each other.
#[pyfunction]
#[pyo3(signature = (photo1, photo2, quality, seed, max_round_trip_error, progress))]
fn correspond(
    py: Python<'_>,
    photo1: PyReadonlyArray3<'_, u8>,
    photo2: PyReadonlyArray3<'_, u8>,
    quality: &str,
    seed: Option<u64>,
    max_round_trip_error: f32,
    progress: Option<Py<PyAny>>,
) -> PyResult<Correspondence> {
    let quality =
        Quality::from_str(quality).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let photo1 = Arc::new(photo_from_array(&photo1).map_err(|err| to_pyerr(py, err))?);
    let photo2 = Arc::new(photo_from_array(&photo2).map_err(|err| to_pyerr(py, err))?);
    let source_size = (photo1.width(), photo1.height());

    let mut builder = pixelmap::Correspondence::builder()
        .quality(quality)
        .max_round_trip_error(max_round_trip_error);
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }

    // The first exception raised by the progress callback, or by a Ctrl-C arriving while
    // the solver runs. `run_with_progress` takes an `FnMut` that cannot fail, so the run
    // goes to completion and the error is raised afterwards; we just stop calling back.
    let mut interrupted: Option<PyErr> = None;

    let result = py.detach(|| {
        builder.run_with_progress(photo1, photo2, |step| {
            if interrupted.is_some() {
                return;
            }
            if let Some(callback) = progress.as_ref() {
                Python::attach(|py| {
                    if let Err(err) = py
                        .check_signals()
                        .and_then(|()| callback.call1(py, (step.step, step.total)).map(|_| ()))
                    {
                        interrupted = Some(err);
                    }
                });
            }
        })
    });

    if let Some(err) = interrupted {
        return Err(err);
    }

    let result = result.map_err(|err| to_pyerr(py, err))?;
    let comparisons = result.comparisons();
    let (forward, backward) = result.into_parts();
    Ok(Correspondence {
        forward,
        backward,
        comparisons,
        source_size,
    })
}

#[pymodule]
fn _pixelmap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Correspondence>()?;
    m.add_function(wrap_pyfunction!(correspond, m)?)?;
    m.add("MIN_DIMENSION", MIN_DIMENSION)?;
    Ok(())
}
