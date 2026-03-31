use faer::{Mat, complex::Complex, traits::RealField};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};
use std::fmt::Display;
use std::fmt::Write;

use crate::{
    array::{FaerArray, FaerArrayMut, FaerArrayRef},
    dtype::FaerDType, numpy::faer_to_numpy,
};

#[pyclass]
pub struct FaerMat {
    pub dtype: FaerDType,
    pub inner: FaerArray,
}

#[pymethods]
impl FaerMat {
    fn __add__(&self, other: &Self) -> PyResult<Self> {
        if self.inner.nrows() != other.inner.nrows() || self.inner.ncols() != other.inner.ncols() {
            return Err(PyValueError::new_err(format!(
                "维度不匹配: ({}, {}) vs ({}, {})",
                self.inner.nrows(),
                self.inner.ncols(),
                other.inner.nrows(),
                other.inner.ncols(),
            )));
        }

        let out_dtype = self.dtype.promote(other.dtype)?;
        let lhs = self.inner.cast_to(out_dtype);
        let rhs = other.inner.cast_to(out_dtype);

        let result = match (lhs.as_ref(), rhs.as_ref()) {
            (FaerArrayRef::F32(a), FaerArrayRef::F32(b)) => FaerArray::F32((a + b).to_owned()),
            (FaerArrayRef::F64(a), FaerArrayRef::F64(b)) => FaerArray::F64((a + b).to_owned()),
            (FaerArrayRef::C32(a), FaerArrayRef::C32(b)) => FaerArray::C32((a + b).to_owned()),
            (FaerArrayRef::C64(a), FaerArrayRef::C64(b)) => FaerArray::C64((a + b).to_owned()),
            _ => unreachable!(),
        };

        Ok(Self {
            dtype: out_dtype,
            inner: result,
        })
    }
    fn __iadd__(&mut self, other: &Self) -> PyResult<()> {
        if self.inner.nrows() != other.inner.nrows() || self.inner.ncols() != other.inner.ncols() {
            return Err(PyValueError::new_err(format!(
                "维度不匹配: ({}, {}) vs ({}, {})",
                self.inner.nrows(),
                self.inner.ncols(),
                other.inner.nrows(),
                other.inner.ncols(),
            )));
        }

        let out_dtype = self.dtype.promote(other.dtype)?;

        if out_dtype == self.dtype {
            // ── 快路径：self 类型已经够用，直接原地 += ──────────────
            let rhs = other.inner.cast_to(out_dtype);
            match (self.inner.as_mut(), rhs.as_ref()) {
                (FaerArrayMut::F32(mut a), FaerArrayRef::F32(b)) => a += b,
                (FaerArrayMut::F64(mut a), FaerArrayRef::F64(b)) => a += b,
                (FaerArrayMut::C32(mut a), FaerArrayRef::C32(b)) => a += b,
                (FaerArrayMut::C64(mut a), FaerArrayRef::C64(b)) => a += b,
                _ => unreachable!(),
            }
        } else {
            // ── 慢路径：self 需要升精度，整体替换 ──────────────────
            // 先把 self 和 other 都 cast 到 out_dtype，相加后写回
            let lhs = self.inner.cast_to(out_dtype);
            let rhs = other.inner.cast_to(out_dtype);
            let result = match (lhs.as_ref(), rhs.as_ref()) {
                (FaerArrayRef::F32(a), FaerArrayRef::F32(b)) => FaerArray::F32((a + b).to_owned()),
                (FaerArrayRef::F64(a), FaerArrayRef::F64(b)) => FaerArray::F64((a + b).to_owned()),
                (FaerArrayRef::C32(a), FaerArrayRef::C32(b)) => FaerArray::C32((a + b).to_owned()),
                (FaerArrayRef::C64(a), FaerArrayRef::C64(b)) => FaerArray::C64((a + b).to_owned()),
                _ => unreachable!(),
            };
            self.inner = result;
            self.dtype = out_dtype;
        }

        Ok(())
    }
    #[classmethod]
    // 默认使用 f32，符合你之前的需求
    #[pyo3(signature = (list, dtype=FaerDType::F32))]
    fn from_list<'py>(
        _cls: Bound<'py, PyType>,
        list: Bound<'py, PyList>,
        dtype: FaerDType,
    ) -> PyResult<Self> {
        let nrows = list.len();
        if nrows == 0 {
            match dtype {
                FaerDType::F32 => {
                    return Ok(Self {
                        inner: FaerArray::F32(Mat::new()),
                        dtype: FaerDType::F32,
                    });
                }
                FaerDType::F64 => {
                    return Ok(Self {
                        inner: FaerArray::F64(Mat::new()),
                        dtype: FaerDType::F64,
                    });
                }
                FaerDType::C32 => {
                    return Ok(Self {
                        inner: FaerArray::C32(Mat::new()),
                        dtype: FaerDType::C32,
                    });
                }
                FaerDType::C64 => {
                    return Ok(Self {
                        inner: FaerArray::C64(Mat::new()),
                        dtype: FaerDType::C64,
                    });
                }
            }
        }

        // 1. 使用 cast_into 代替 downcast_into
        // 注意：get_item 返回的是 Bound<'py, PyAny>
        let first_row = list
            .get_item(0)?
            .cast_into::<PyList>() // 最新 API
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a list of lists")
            })?;

        let ncols = first_row.len();

        // 根据逻辑类型分发到具体的构造过程
        match dtype {
            FaerDType::F32 => {
                let mut mat = Mat::<f32>::zeros(nrows, ncols);
                for i in 0..nrows {
                    let row = list.get_item(i)?.cast_into::<PyList>()?;
                    for j in 0..ncols {
                        mat[(i, j)] = row.get_item(j)?.extract::<f32>()?;
                    }
                }
                Ok(Self {
                    inner: FaerArray::F32(mat),
                    dtype: FaerDType::F32,
                })
            }
            FaerDType::F64 => {
                let mut mat = Mat::<f64>::zeros(nrows, ncols);
                for i in 0..nrows {
                    let row = list.get_item(i)?.cast_into::<PyList>()?;
                    for j in 0..ncols {
                        mat[(i, j)] = row.get_item(j)?.extract::<f64>()?;
                    }
                }
                Ok(Self {
                    inner: FaerArray::F64(mat),
                    dtype: FaerDType::F64,
                })
            }
            FaerDType::C32 => {
                let mut mat = Mat::<Complex<f32>>::zeros(nrows, ncols);
                for i in 0..nrows {
                    let row = list.get_item(i)?.cast_into::<PyList>()?;
                    for j in 0..ncols {
                        let c: Complex<f32> = row.get_item(j)?.extract()?;

                        // 步骤 2: 将数据填入 faer::complex::Complex
                        // 注意：faer 的 Complex::new 接收 (re, im)
                        mat[(i, j)] = Complex::new(c.re, c.im);
                    }
                }
                Ok(Self {
                    inner: FaerArray::C32(mat),
                    dtype: FaerDType::C32,
                })
            }
            FaerDType::C64 => {
                let mut mat = Mat::<Complex<f64>>::zeros(nrows, ncols);
                for i in 0..nrows {
                    let row = list.get_item(i)?.cast_into::<PyList>()?;
                    for j in 0..ncols {
                        // 步骤 1: 提取为 PyO3 认识的 num_complex::Complex64
                        let c: Complex<f64> = row.get_item(j)?.extract()?;

                        // 步骤 2: 将数据填入 faer::complex::Complex
                        // 注意：faer 的 Complex::new 接收 (re, im)
                        mat[(i, j)] = Complex::new(c.re, c.im);
                    }
                }
                Ok(Self {
                    inner: FaerArray::C64(mat),
                    dtype: FaerDType::C64,
                })
            }
        }
    }
    #[getter]
    fn dtype(&self) -> FaerDType {
        self.dtype
    }
    // 实现 shape 的多类型分发
    #[getter]
    fn shape(&self) -> (usize, usize) {
        match &self.inner {
            FaerArray::F32(m) => (m.nrows(), m.ncols()),
            FaerArray::F64(m) => (m.nrows(), m.ncols()),
            FaerArray::C32(m) => (m.nrows(), m.ncols()),
            FaerArray::C64(m) => (m.nrows(), m.ncols()),
        }
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = self.shape();
        let dtype_obj = self.dtype();
        // 2. 现在再从这个存活的对象里借用 name
        let dtype_name = dtype_obj.name();

        // 基础信息头
        let mut s = format!(
            "FaerMat(shape=({}, {}), dtype={})\n",
            rows, cols, dtype_name
        );

        // 根据内部类型分发打印逻辑
        let content = match &self.inner {
            FaerArray::F32(m) => format_real_mat(m),
            FaerArray::F64(m) => format_real_mat(m),
            FaerArray::C32(m) => format_complex_mat(m),
            FaerArray::C64(m) => format_complex_mat(m),
        };

        s.push_str(&content);
        s
    }
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            FaerArray::F32(m) => faer_to_numpy(py, m),
            FaerArray::F64(m) => faer_to_numpy(py, m),
            FaerArray::C32(m) => faer_to_numpy(py, m),
            FaerArray::C64(m) => faer_to_numpy(py, m),
        }
    }
}
/// 格式化实数矩阵 (f32/f64)
// 使用 RealField 替代 Entity
fn format_real_mat<T: Display + RealField>(mat: &faer::Mat<T>) -> String {
    let mut buf = String::from(" [");
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    for i in 0..nrows {
        if i > 0 {
            buf.push_str("  ");
        }
        buf.push_str("[");
        for j in 0..ncols {
            let val = &mat[(i, j)];
            write!(buf, "{:>8.4}", val).unwrap(); // >8 表示占8位靠右，对齐更美观
            if j < ncols - 1 {
                buf.push_str(", ");
            }
        }
        buf.push_str("]");
        if i < nrows - 1 {
            buf.push_str("\n");
        }
    }
    buf.push_str("]");
    buf
}

/// 格式化复数矩阵 (c32/c64)
// 注意：faer 的 Complex<T> 里的 T 必须是 RealField
fn format_complex_mat<T: Display + RealField>(
    mat: &faer::Mat<faer::complex::Complex<T>>,
) -> String {
    let mut buf = String::from(" [");
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    for i in 0..nrows {
        if i > 0 {
            buf.push_str("  ");
        }
        buf.push_str("[");
        for j in 0..ncols {
            let c = &mat[(i, j)];
            // 格式化虚部时使用 + 强制显示正负号
            write!(buf, "({:>7.4}{:+.4}j)", c.re, c.im).unwrap();
            if j < ncols - 1 {
                buf.push_str(", ");
            }
        }
        buf.push_str("]");
        if i < nrows - 1 {
            buf.push_str("\n");
        }
    }
    buf.push_str("]");
    buf
}
