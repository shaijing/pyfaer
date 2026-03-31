use pyo3::pyclass;

use crate::{array::FaerArray, dtype::FaerDType};

#[pyclass]
pub struct FaerMat {
    pub dtype: FaerDType,
    pub inner: FaerArray,
}
