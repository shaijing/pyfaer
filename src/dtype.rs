use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use std::collections::HashMap;
use std::sync::LazyLock; // 引入标准库的 LazyLock
#[pyclass(eq, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaerDType {
    // U32,
    // U64,
    // I32,
    // I64,
    F32,
    F64,
    C32,
    C64,
}

impl Default for FaerDType {
    fn default() -> Self {
        FaerDType::F32
    }
}

/// 全集，用于格搜索
const ALL_TYPES: [FaerDType; 4] = [
    // FaerDType::U32,
    // FaerDType::U64,
    // FaerDType::I32,
    // FaerDType::I64,
    FaerDType::F32,
    FaerDType::F64,
    FaerDType::C32,
    FaerDType::C64,
];
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capability {
    pub is_complex: bool,
    pub is_float: bool,
    pub is_signed: bool,
    /// 存储位宽（f32=32, f64=64, i32=32 ...）
    pub bits: u32,
    ///   整数类型 = bits - if signed { 1 } else { 0 }
    ///   f32      = 24  (1 隐含位 + 23 尾数位)
    ///   f64      = 53  (1 隐含位 + 52 尾数位)
    ///   复数同其实部
    pub int_mantissa_bits: u32,
}
impl Capability {
    /// 偏序关系：self ≤ other（other 能无损表示 self 的所有值）
    pub fn is_subset_of(&self, other: &Self) -> bool {
        // 域约束：复数 > 浮点 > 整数，只能向上兼容
        if self.is_complex && !other.is_complex {
            return false;
        }
        if self.is_float && !other.is_float {
            return false;
        }

        // 符号约束：有符号整数含负数，无符号 other 无法表示
        if self.is_signed && !other.is_signed {
            return false;
        }

        // 整数范围约束：other 的尾数位必须 >= self 的无损整数位
        //   U64(64) → F64(53)：53 < 64，失败 ✓（F64 无法精确表示所有 u64）
        //   U32(32) → F64(53)：53 >= 32，通过 ✓
        //   U32(32) → I64(63)：63 >= 32，通过 ✓（无符号→有符号，位数够就行）
        //   I32(31) → U32：有符号→无符号，上面已经拦截
        if other.int_mantissa_bits < self.int_mantissa_bits {
            return false;
        }
        true
    }
}
impl FaerDType {
    /// 映射到能力向量（方法二的思路，但附在 enum 上）
    pub fn capability(self) -> Capability {
        match self {
            //                               cmplx  float  sign   bits  mantissa
            // Self::U32 => Capability {
            //     is_complex: false,
            //     is_float: false,
            //     is_signed: false,
            //     bits: 32,
            //     int_mantissa_bits: 32,
            // },
            // Self::U64 => Capability {
            //     is_complex: false,
            //     is_float: false,
            //     is_signed: false,
            //     bits: 64,
            //     int_mantissa_bits: 64,
            // },
            // Self::I32 => Capability {
            //     is_complex: false,
            //     is_float: false,
            //     is_signed: true,
            //     bits: 32,
            //     int_mantissa_bits: 31,
            // },
            // Self::I64 => Capability {
            //     is_complex: false,
            //     is_float: false,
            //     is_signed: true,
            //     bits: 64,
            //     int_mantissa_bits: 63,
            // },
            Self::F32 => Capability {
                is_complex: false,
                is_float: true,
                is_signed: true,
                bits: 32,
                int_mantissa_bits: 24,
            },
            Self::F64 => Capability {
                is_complex: false,
                is_float: true,
                is_signed: true,
                bits: 64,
                int_mantissa_bits: 53,
            },
            Self::C32 => Capability {
                is_complex: true,
                is_float: true,
                is_signed: true,
                bits: 32,
                int_mantissa_bits: 24,
            },
            Self::C64 => Capability {
                is_complex: true,
                is_float: true,
                is_signed: true,
                bits: 64,
                int_mantissa_bits: 53,
            },
        }
    }

    /// 成本函数，决定"最小"上界（方法二 cost 字段 → 方法一的方法）
    pub fn cost(self) -> u32 {
        let cap = self.capability();
        cap.bits
            + if cap.is_complex {
                1000
            } else if cap.is_float {
                500
            } else {
                0
            }
    }

    /// 核心 join：O(N) 格搜索（仅在预热阶段调用一次）
    fn promote_inner(a: Self, b: Self) -> Option<Self> {
        let (ca, cb) = (a.capability(), b.capability());
        ALL_TYPES
            .iter()
            .copied()
            .filter(|&t| {
                let ct = t.capability();
                ca.is_subset_of(&ct) && cb.is_subset_of(&ct)
            })
            .min_by_key(|t| t.cost())
    }
}

static PROMOTION_TABLE: LazyLock<HashMap<(FaerDType, FaerDType), Option<FaerDType>>> =
    LazyLock::new(|| {
        let mut m = HashMap::with_capacity(64);
        for &a in &ALL_TYPES {
            for &b in &ALL_TYPES {
                m.insert((a, b), FaerDType::promote_inner(a, b));
            }
        }
        m
    });
// static PROMOTION_TABLE: LazyLock<HashMap<(FaerDType, FaerDType), FaerDType>> =
//     LazyLock::new(|| {
//         let mut m = HashMap::with_capacity(64);
//         for &a in &ALL_TYPES {
//             for &b in &ALL_TYPES {
//                 m.insert((a, b), FaerDType::promote_inner(a, b));
//             }
//         }
//         m
//     });

#[pymethods]
impl FaerDType {
    /// Python: FaerDType.F32.promote(FaerDType.C32) -> FaerDType.C32
    pub fn promote(&self, other: FaerDType) -> PyResult<FaerDType> {
        // *PROMOTION_TABLE.get(&(*self, other)).unwrap()
        PROMOTION_TABLE
            .get(&(*self, other))
            .copied()
            .flatten()
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "无法无损提升 {:?} 和 {:?}：不存在公共上界",
                    self, other
                ))
            })
    }

    pub fn is_complex(&self) -> bool {
        self.capability().is_complex
    }
    pub fn is_float(&self) -> bool {
        self.capability().is_float
    }
    pub fn is_signed(&self) -> bool {
        self.capability().is_signed
    }
    pub fn bits(&self) -> u32 {
        self.capability().bits
    }
    pub fn name(&self) -> &'static str {
        match self {
            // Self::U32 => "u32",
            // Self::U64 => "u64",
            // Self::I32 => "i32",
            // Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C32 => "c32",
            Self::C64 => "c64",
        }
    }
    pub fn __repr__(&self) -> &'static str {
        match self {
            // Self::U32 => "u32",
            // Self::U64 => "u64",
            // Self::I32 => "i32",
            // Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C32 => "c32",
            Self::C64 => "c64",
        }
    }
}
impl std::str::FromStr for FaerDType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            // "u32" => Ok(Self::U32),
            // "u64" => Ok(Self::U64),
            // "i32" => Ok(Self::I32),
            // "i64" => Ok(Self::I64),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "c32" | "complex64" => Ok(Self::C32),
            "c64" | "complex128" => Ok(Self::C64),
            _ => Err(format!("unknown dtype '{s}'")),
        }
    }
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
    use std::str::FromStr;

    use super::*;

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(FaerDType::from_str("f32").unwrap(), FaerDType::F32);
        assert_eq!(FaerDType::from_str("f64").unwrap(), FaerDType::F64);
        assert_eq!(FaerDType::from_str("c32").unwrap(), FaerDType::C32);
        assert_eq!(FaerDType::from_str("c64").unwrap(), FaerDType::C64);
    }


    #[test]
    fn test_promotion_f32() {
        assert_eq!(
            FaerDType::F32.promote(FaerDType::F32).unwrap(),
            FaerDType::F32
        );
        assert_eq!(
            FaerDType::F32.promote(FaerDType::F64).unwrap(),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::F32.promote(FaerDType::C32).unwrap(),
            FaerDType::C32
        );
        assert_eq!(
            FaerDType::F32.promote(FaerDType::C64).unwrap(),
            FaerDType::C64
        );
    }

    #[test]
    fn test_promotion_f64() {
        assert_eq!(
            FaerDType::F64.promote(FaerDType::F64).unwrap(),
            FaerDType::F64
        );
        assert_eq!(
            FaerDType::F64.promote(FaerDType::C32).unwrap(),
            FaerDType::C64
        );
        assert_eq!(
            FaerDType::F64.promote(FaerDType::C64).unwrap(),
            FaerDType::C64
        );
    }
}
