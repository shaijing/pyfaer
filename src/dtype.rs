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
    fn __repr__(&self) -> &str {
        match self {
            Self::U32 => "faer.usize32",
            Self::U64 => "faer.usize64",
            Self::I32 => "faer.int32",
            Self::I64 => "faer.int64",
            Self::F32 => "faer.float32",
            Self::F64 => "faer.float64",
            Self::C32 => "faer.complex64",
            Self::C64 => "faer.complex128",
        }
    }
    #[staticmethod]
    pub fn promote(a: Self, b: Self) -> Self {
        // 2. 内部逻辑直接调用 Rust 引用方法，避免多次拷贝
        let ia = a.get_info_internal();
        let ib = b.get_info_internal();
        // 1. 确定基础属性需求
        let mut target_complex = ia.complex || ib.complex;
        let mut target_float = ia.float || ib.float;
        let target_signed = ia.signed || ib.signed;

        // 2. 计算需要的最小位宽
        // 如果一个是无符号，一个是有符号，且位宽相同，我们需要更多位
        // 修改后的逻辑：
        let min_bits = std::cmp::max(ia.bits, ib.bits);
        let sign_conflict = ia.signed != ib.signed;

        // 只有当最大的位宽(比如64)依然无法容纳另一种符号时，才考虑 float
        let mut target_float = ia.float || ib.float;
        if !target_float && sign_conflict && min_bits == 64 && (ia.bits == 64 || ib.bits == 64) {
            target_float = true;
        }
        FAER_DTYPE_INFOS
            .iter()
            // .filter(|cand| {
            //     (ia.complex <= cand.complex && ib.complex <= cand.complex)
            //         && (ia.float <= cand.float && ib.float <= cand.float)
            //         && (ia.signed <= cand.signed && ib.signed <= cand.signed)
            //         && (ia.bits <= cand.bits && ib.bits <= cand.bits)
            // })
            .filter(|cand| {
                // 严格遵守 complex/float 的单向提升（除非上面强制转了 float）
                (target_complex <= cand.complex)
                && (target_float <= cand.float)
                && (target_signed <= cand.signed)
                // 如果是浮点数，bits 的比较逻辑通常允许 F64 承接 I64
                && (cand.bits >= std::cmp::min(min_bits, 64))
            })
            .min_by_key(|cand| cost(cand))
            .map(|i| i.dtype)
            .unwrap_or(FaerDType::C64)
    }
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
