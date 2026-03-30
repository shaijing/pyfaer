use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
#[pyclass(eq, skip_from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum FaerDType {
    F32,
    F64,
    C32,
    C64,
}

impl<'a, 'py> FromPyObject<'a, 'py> for FaerDType {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        // 尝试作为 FaerDType 实例
        if let Ok(v) = ob.cast::<FaerDType>() {
            return Ok(*v.borrow());
        }
        // 尝试从字符串构造
        if let Ok(s) = ob.cast::<pyo3::types::PyString>() {
            return match s.to_str()? {
                "float32" | "f32" => Ok(FaerDType::F32),
                "float64" | "f64" => Ok(FaerDType::F64),
                "complex64" | "c32" => Ok(FaerDType::C32),
                "complex128" | "c64" => Ok(FaerDType::C64),
                other => Err(PyValueError::new_err(format!("未知 dtype: {other:?}"))),
            };
        }
        Err(PyTypeError::new_err("dtype 必须是 FaerDType 实例或字符串"))
    }
}
#[pymethods]
impl FaerDType {
    fn __repr__(&self) -> &str {
        match self {
            Self::F32 => "faer.float32",
            Self::F64 => "faer.float64",
            Self::C32 => "faer.complex64",
            Self::C64 => "faer.complex128",
        }
    }

    // 增加一个辅助方法，方便用户获取字符串名
    #[getter]
    pub fn name(&self) -> &str {
        match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::C32 => "complex64",
            Self::C64 => "complex128",
        }
    }
    #[staticmethod]
    pub fn promote(a: FaerDType, b: FaerDType) -> Self {
        match (a, b) {
            (Self::F32, Self::F32) => Self::F32,
            (Self::F64, Self::F64) => Self::F64,
            (Self::C32, Self::C32) => Self::C32,
            (Self::C64, Self::C64) => Self::C64,

            // 精度提升
            (Self::F32, Self::F64) | (Self::F64, Self::F32) => Self::F64,
            (Self::C32, Self::C64) | (Self::C64, Self::C32) => Self::C64,

            // 实数 → 复数
            (Self::F32, Self::C32) | (Self::C32, Self::F32) => Self::C32,
            (Self::F64, Self::C64) | (Self::C64, Self::F64) => Self::C64,

            // 混合提升
            (Self::F32, Self::C64) | (Self::C64, Self::F32) => Self::C64,
            (Self::F64, Self::C32) | (Self::C32, Self::F64) => Self::C64,
        }
    }
}
