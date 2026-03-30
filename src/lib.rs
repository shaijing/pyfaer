use pyo3::prelude::*;
/// A Python module implemented in Rust.
#[pymodule]
mod _faer {
    use faer::Mat;
    use numpy::PyReadonlyArray2;
    use numpy::{PyArray2, PyArrayMethods};
    use pyo3::{
        exceptions::PyValueError,
        prelude::*,
        types::{PyList, PyType},
    };
    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
    #[pyclass]
    pub struct FaerMatf64 {
        // 内部持有 faer 的实矩阵
        pub inner: Mat<f64>,
    }
    #[pymethods]
    impl FaerMatf64 {
        // 1. 构造函数：创建一个全零矩阵
        #[new]
        fn new(nrows: usize, ncols: usize) -> Self {
            Self {
                inner: Mat::<f64>::zeros(nrows, ncols),
            }
        }

        // 2. 获取维度
        #[getter]
        fn shape(&self) -> (usize, usize) {
            (self.inner.nrows(), self.inner.ncols())
        }

        // 3. 与 NumPy 交互：将 faer 矩阵转换为 NumPy 数组 (通常涉及一次拷贝)
        fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            let (nrows, ncols) = (self.inner.nrows(), self.inner.ncols());

            // 1. 创建数组 (需要 unsafe)
            let py_array = unsafe { PyArray2::<f64>::new(py, [nrows, ncols], false) };

            // 2. 在 0.28 中，Bound<'_, PyArray2> 直接拥有 readonly() 和 rw() 的替代者
            // 如果 .writable() 报错，尝试直接使用 .as_array_mut()
            // 因为 PyArrayMethods trait 已经为 Bound<'_, PyArray> 实现了这个方法
            let mut array_view = unsafe { py_array.as_array_mut() };

            let faer_view = self.inner.as_ref();

            // 3. 填充数据
            for i in 0..nrows {
                for j in 0..ncols {
                    // 使用 ndarray 的风格进行赋值
                    array_view[[i, j]] = faer_view[(i, j)];
                }
            }

            py_array
        }
        // 4. 示例计算：计算矩阵的迹 (Trace)
        fn trace(&self) -> f64 {
            // faer 的 API 非常直观
            self.inner.as_ref().diagonal().column_vector().sum()
        }

        #[classmethod]
        fn from_list<'py>(_cls: Bound<'py, PyType>, list: Bound<'py, PyList>) -> PyResult<Self> {
            let nrows = list.len();
            if nrows == 0 {
                return Ok(Self { inner: Mat::new() });
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
            let mut mat = Mat::<f64>::zeros(nrows, ncols);

            for i in 0..nrows {
                let row_item = list
                    .get_item(i)?
                    .cast_into::<PyList>() // 最新 API
                    .map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Row {} is not a list",
                            i
                        ))
                    })?;

                if row_item.len() != ncols {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Inconsistent row lengths",
                    ));
                }

                for j in 0..ncols {
                    // 2. extract 依然可用，它是 BoundMethods 的一部分
                    let val: f64 = row_item.get_item(j)?.extract()?;
                    mat[(i, j)] = val;
                }
            }

            Ok(Self { inner: mat })
        }
        #[classmethod]
        fn from_numpy<'py>(
            _cls: Bound<'py, PyType>,
            array: PyReadonlyArray2<'py, f64>,
        ) -> PyResult<Self> {
            // 在新版本中，我们可以直接从 PyReadonlyArray2 获取 ndarray 视图
            let view = array.as_array();
            let (nrows, ncols) = (view.shape()[0], view.shape()[1]);

            // 构造 faer 矩阵
            let mut mat = Mat::<f64>::zeros(nrows, ncols);

            // 这里的赋值逻辑依然有效
            for i in 0..nrows {
                for j in 0..ncols {
                    mat[(i, j)] = view[[i, j]];
                }
            }

            Ok(Self { inner: mat })
        }
        // 实现 A + B
        fn __add__(&self, other: &Self) -> PyResult<Self> {
            // 1. 维度检查
            if self.inner.nrows() != other.inner.nrows()
                || self.inner.ncols() != other.inner.ncols()
            {
                return Err(PyValueError::new_err(format!(
                    "Matrix dimensions must match for addition: ({}, {}) vs ({}, {})",
                    self.inner.nrows(),
                    self.inner.ncols(),
                    other.inner.nrows(),
                    other.inner.ncols()
                )));
            }

            // 2. 执行加法
            // faer 的 Mat 实现了加法运算符，它会自动处理并行化（取决于编译配置）
            let result_inner = &self.inner + &other.inner;

            // 3. 返回新的包装类
            Ok(Self {
                inner: result_inner,
            })
        }
        fn __matmul__(&self, other: &Self) -> PyResult<Self> {
            // 1. 检查矩阵乘法维度规则: (M x K) @ (K x N)
            if self.inner.ncols() != other.inner.nrows() {
                return Err(PyValueError::new_err(format!(
                    "Dimension mismatch for matrix multiplication: columns of A ({}) must match rows of B ({})",
                    self.inner.ncols(),
                    other.inner.nrows()
                )));
            }

            // 2. 调用 faer 的高性能乘法
            // Parallelism::Rayon(0) 会自动利用所有可用的 CPU 核心
            let result_inner = &self.inner * &other.inner;

            Ok(Self {
                inner: result_inner,
            })
        }
    }
}
