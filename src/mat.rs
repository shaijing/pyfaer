use faer::Mat;
use faer::complex::Complex;
use faer::traits::RealField;
use pyo3::{
    prelude::*,
    types::{PyList, PyType},
};
use std::fmt::Display;
use std::fmt::Write;

use crate::dtype::FaerDType;

pub enum FaerArray {
    F32(Mat<f32>),
    F64(Mat<f64>),
    C32(Mat<Complex<f32>>), // 复数 32 位 (float x 2)
    C64(Mat<Complex<f64>>), // 复数 64 位 (double x 2)
}

#[pyclass]
pub struct FaerMat {
    pub inner: FaerArray,
}
#[pymethods]
impl FaerMat {
    #[getter]
    fn dtype(&self) -> FaerDType {
        match &self.inner {
            FaerArray::F32(_) => FaerDType::F32,
            FaerArray::F64(_) => FaerDType::F64,
            FaerArray::C32(_) => FaerDType::C32,
            FaerArray::C64(_) => FaerDType::C64,
        }
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
            return Ok(Self {
                inner: FaerArray::F32(Mat::new()),
            });
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
        let inner = match dtype {
            FaerDType::F32 => {
                let mut mat = Mat::<f32>::zeros(nrows, ncols);
                for i in 0..nrows {
                    let row = list.get_item(i)?.cast_into::<PyList>()?;
                    for j in 0..ncols {
                        mat[(i, j)] = row.get_item(j)?.extract::<f32>()?;
                    }
                }
                FaerArray::F32(mat)
            }
            FaerDType::F64 => {
                let mut mat = Mat::<f64>::zeros(nrows, ncols);
                for i in 0..nrows {
                    let row = list.get_item(i)?.cast_into::<PyList>()?;
                    for j in 0..ncols {
                        mat[(i, j)] = row.get_item(j)?.extract::<f64>()?;
                    }
                }
                FaerArray::F64(mat)
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
                FaerArray::C32(mat)
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
                FaerArray::C64(mat)
            }
        };

        Ok(Self { inner })
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

// #[pyclass]
// pub struct FaerMatf64 {
//     // 内部持有 faer 的实矩阵
//     pub inner: Mat<f64>,
// }
// #[pymethods]
// impl FaerMatf64 {
//     // 1. 构造函数：创建一个全零矩阵
//     #[new]
//     fn new(nrows: usize, ncols: usize) -> Self {
//         Self {
//             inner: Mat::<f64>::zeros(nrows, ncols),
//         }
//     }
//     #[staticmethod]
//     fn zeros(nrows: usize, ncols: usize) -> Self {
//         Self {
//             inner: Mat::<f64>::zeros(nrows, ncols),
//         }
//     }
//     #[staticmethod]
//     fn ones(nrows: usize, ncols: usize) -> Self {
//         Self {
//             inner: Mat::<f64>::ones(nrows, ncols),
//         }
//     }
//     #[staticmethod]
//     fn identity(nrows: usize, ncols: usize) -> Self {
//         Self {
//             inner: Mat::<f64>::identity(nrows, ncols),
//         }
//     }
//     #[staticmethod]
//     fn full(nrows: usize, ncols: usize, value: f64) -> Self {
//         Self {
//             inner: Mat::<f64>::full(nrows, ncols, value),
//         }
//     }
//     // 2. 获取维度
//     #[getter]
//     fn shape(&self) -> (usize, usize) {
//         (self.inner.nrows(), self.inner.ncols())
//     }

//     // 3. 与 NumPy 交互：将 faer 矩阵转换为 NumPy 数组 (通常涉及一次拷贝)
//     fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
//         let (nrows, ncols) = (self.inner.nrows(), self.inner.ncols());

//         // 1. 创建数组 (需要 unsafe)
//         let py_array = unsafe { PyArray2::<f64>::new(py, [nrows, ncols], false) };

//         // 2. 在 0.28 中，Bound<'_, PyArray2> 直接拥有 readonly() 和 rw() 的替代者
//         // 如果 .writable() 报错，尝试直接使用 .as_array_mut()
//         // 因为 PyArrayMethods trait 已经为 Bound<'_, PyArray> 实现了这个方法
//         let mut array_view = unsafe { py_array.as_array_mut() };

//         let faer_view = self.inner.as_ref();

//         // 3. 填充数据
//         for i in 0..nrows {
//             for j in 0..ncols {
//                 // 使用 ndarray 的风格进行赋值
//                 array_view[[i, j]] = faer_view[(i, j)];
//             }
//         }

//         py_array
//     }
//     // 4. 示例计算：计算矩阵的迹 (Trace)
//     fn trace(&self) -> f64 {
//         // faer 的 API 非常直观
//         self.inner.as_ref().diagonal().column_vector().sum()
//     }

//     #[classmethod]
//     fn from_list<'py>(_cls: Bound<'py, PyType>, list: Bound<'py, PyList>) -> PyResult<Self> {
//         let nrows = list.len();
//         if nrows == 0 {
//             return Ok(Self { inner: Mat::new() });
//         }

//         // 1. 使用 cast_into 代替 downcast_into
//         // 注意：get_item 返回的是 Bound<'py, PyAny>
//         let first_row = list
//             .get_item(0)?
//             .cast_into::<PyList>() // 最新 API
//             .map_err(|_| {
//                 PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a list of lists")
//             })?;

//         let ncols = first_row.len();
//         let mut mat = Mat::<f64>::zeros(nrows, ncols);

//         for i in 0..nrows {
//             let row_item = list
//                 .get_item(i)?
//                 .cast_into::<PyList>() // 最新 API
//                 .map_err(|_| {
//                     PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
//                         "Row {} is not a list",
//                         i
//                     ))
//                 })?;

//             if row_item.len() != ncols {
//                 return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                     "Inconsistent row lengths",
//                 ));
//             }

//             for j in 0..ncols {
//                 // 2. extract 依然可用，它是 BoundMethods 的一部分
//                 let val: f64 = row_item.get_item(j)?.extract()?;
//                 mat[(i, j)] = val;
//             }
//         }

//         Ok(Self { inner: mat })
//     }
//     #[classmethod]
//     fn from_numpy<'py>(
//         _cls: Bound<'py, PyType>,
//         array: PyReadonlyArray2<'py, f64>,
//     ) -> PyResult<Self> {
//         // 在新版本中，我们可以直接从 PyReadonlyArray2 获取 ndarray 视图
//         let view = array.as_array();
//         let (nrows, ncols) = (view.shape()[0], view.shape()[1]);

//         // 构造 faer 矩阵
//         let mut mat = Mat::<f64>::zeros(nrows, ncols);

//         // 这里的赋值逻辑依然有效
//         for i in 0..nrows {
//             for j in 0..ncols {
//                 mat[(i, j)] = view[[i, j]];
//             }
//         }

//         Ok(Self { inner: mat })
//     }
//     // 实现 A + B
//     fn __add__(&self, other: &Self) -> PyResult<Self> {
//         // 1. 维度检查
//         if self.inner.nrows() != other.inner.nrows() || self.inner.ncols() != other.inner.ncols() {
//             return Err(PyValueError::new_err(format!(
//                 "Matrix dimensions must match for addition: ({}, {}) vs ({}, {})",
//                 self.inner.nrows(),
//                 self.inner.ncols(),
//                 other.inner.nrows(),
//                 other.inner.ncols()
//             )));
//         }

//         // 2. 执行加法
//         // faer 的 Mat 实现了加法运算符，它会自动处理并行化（取决于编译配置）
//         let result_inner = &self.inner + &other.inner;

//         // 3. 返回新的包装类
//         Ok(Self {
//             inner: result_inner,
//         })
//     }
//     fn __matmul__(&self, other: &Self) -> PyResult<Self> {
//         // 1. 检查矩阵乘法维度规则: (M x K) @ (K x N)
//         if self.inner.ncols() != other.inner.nrows() {
//             return Err(PyValueError::new_err(format!(
//                 "Dimension mismatch for matrix multiplication: columns of A ({}) must match rows of B ({})",
//                 self.inner.ncols(),
//                 other.inner.nrows()
//             )));
//         }

//         // 2. 调用 faer 的高性能乘法
//         // Parallelism::Rayon(0) 会自动利用所有可用的 CPU 核心
//         let result_inner = &self.inner * &other.inner;

//         Ok(Self {
//             inner: result_inner,
//         })
//     }
// }
