use pyo3::prelude::*;
mod array;
mod cast;
mod dtype;
mod mat;
mod numpy;
mod ufunc;
/// A Python module implemented in Rust.
#[pymodule]
mod _faer {
    #[pymodule_export]
    use crate::dtype::FaerDType;
    #[pymodule_export]
    use crate::mat::FaerMat;
    use pyo3::prelude::*;
    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
