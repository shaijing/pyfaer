use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

#[pyclass(eq, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaerDType {
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    C32,
    C64,
}

#[derive(Clone, Copy)]
pub struct FaerDTypeInfo {
    pub dtype: FaerDType,
    pub signed: bool,
    pub float: bool,
    pub complex: bool,
    pub bits: u32,
}

const FAER_DTYPE_INFOS: &[FaerDTypeInfo] = &[
    FaerDTypeInfo {
        dtype: FaerDType::U32,
        signed: false,
        float: false,
        complex: false,
        bits: 32,
    },
    FaerDTypeInfo {
        dtype: FaerDType::U64,
        signed: false,
        float: false,
        complex: false,
        bits: 64,
    },
    FaerDTypeInfo {
        dtype: FaerDType::I32,
        signed: true,
        float: false,
        complex: false,
        bits: 32,
    },
    FaerDTypeInfo {
        dtype: FaerDType::I64,
        signed: true,
        float: false,
        complex: false,
        bits: 64,
    },
    FaerDTypeInfo {
        dtype: FaerDType::F32,
        signed: true,
        float: true,
        complex: false,
        bits: 32,
    },
    FaerDTypeInfo {
        dtype: FaerDType::F64,
        signed: true,
        float: true,
        complex: false,
        bits: 64,
    },
    FaerDTypeInfo {
        dtype: FaerDType::C32,
        signed: true,
        float: true,
        complex: true,
        bits: 32,
    },
    FaerDTypeInfo {
        dtype: FaerDType::C64,
        signed: true,
        float: true,
        complex: true,
        bits: 64,
    },
];
#[pymethods]
impl FaerDType {
    #[staticmethod]
    pub fn promote(a: Self, b: Self) -> Self {}
}
#[inline]
fn cost(info: &FaerDTypeInfo) -> u32 {
    info.bits + (if info.float { 100 } else { 0 }) + (if info.complex { 200 } else { 0 })
    // info.bits * 10 + (if info.float { 1000 } else { 0 }) + (if info.complex { 2000 } else { 0 })
}
// 独立的 impl 块，不带 #[pymethods]，用于存放返回引用的 Rust 内部函数
impl FaerDType {
    #[inline]
    fn get_info_internal(&self) -> &'static FaerDTypeInfo {
        FAER_DTYPE_INFOS
            .iter()
            .find(|i| i.dtype == *self)
            .expect("DType info must exist")
    }
    // #[inline]
    // fn from_str(s: &str) -> Result<Self, PyErr> {
    //     match s {
    //         "float32" | "f32" => Ok(FaerDType::F32),
    //         "float64" | "f64" => Ok(FaerDType::F64),
    //         "complex64" | "c32" => Ok(FaerDType::C32),
    //         "complex128" | "c64" => Ok(FaerDType::C64),
    //         other => Err(PyValueError::new_err(format!("未知 dtype: {other:?}"))),
    //     }
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
        // if let Ok(name_attr) = ob.getattr("name").or_else(|_| ob.getattr("__name__")) {
        //     if let Ok(s) = name_attr.extract::<&str>() {
        //         // 过滤掉可能存在的 numpy 前缀或后缀
        //         return Self::from_str(s);
        //     }
        // }
        Err(PyTypeError::new_err("dtype 必须是 FaerDType 实例或字符串"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dtype_promotion_i32() {
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::I32),
            FaerDType::I32
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::I64),
            FaerDType::I64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::U32),
            FaerDType::I64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::U64),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::F32),
            FaerDType::F32
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::F64),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::C32),
            FaerDType::C32
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I32, FaerDType::C64),
            FaerDType::C64
        );
    }
    #[test]
    fn test_dtype_promotion_i64() {
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::I64),
            FaerDType::I64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::U32),
            FaerDType::I64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::U64),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::F32),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::F64),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::C32),
            FaerDType::C64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::I64, FaerDType::C64),
            FaerDType::C64
        );
    }
    #[test]
    fn test_dtype_promotion() {
        assert_eq!(
            FaerDType::promote(FaerDType::U32, FaerDType::U64),
            FaerDType::U64
        );

        assert_eq!(
            FaerDType::promote(FaerDType::F32, FaerDType::F32),
            FaerDType::F32
        );
        assert_eq!(
            FaerDType::promote(FaerDType::C64, FaerDType::C64),
            FaerDType::C64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::F32, FaerDType::F64),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::F32, FaerDType::C32),
            FaerDType::C32
        );
        assert_eq!(
            FaerDType::promote(FaerDType::F64, FaerDType::C32),
            FaerDType::C64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::F64, FaerDType::C64),
            FaerDType::C64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::C32, FaerDType::C64),
            FaerDType::C64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::U32, FaerDType::I32),
            FaerDType::I64
        );
        assert_eq!(
            FaerDType::promote(FaerDType::U32, FaerDType::F32),
            FaerDType::F32
        );

        assert_eq!(
            FaerDType::promote(FaerDType::U32, FaerDType::C32),
            FaerDType::C32
        );
    }
}
