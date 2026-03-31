use faer::Mat;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::{Bound, PyAny, PyResult, Python};

pub fn faer_to_numpy<'py, T>(py: Python<'py>, m: &Mat<T>) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Copy,
{
    let nrows = m.nrows();
    let ncols = m.ncols();

    // 直接分配一块 numpy 管理的内存，列优先（Fortran order）
    // 和 faer 默认布局一致，避免转置拷贝
    let arr = unsafe {
        PyArray2::<T>::new(py, [nrows, ncols], true /* fortran order */)
    };

    // 逐列把 faer 的数据 memcpy 进 numpy 的缓冲区
    for j in 0..ncols {
        let src = m.col_as_slice(j); // faer 列切片，&[T]
        let dst = unsafe {
            // get_raw_data 返回 *mut T，偏移到第 j 列的起始位置
            // Fortran order 下第 j 列起始偏移 = j * nrows
            std::slice::from_raw_parts_mut(arr.data().add(j * nrows), nrows)
        };
        dst.copy_from_slice(src);
    }

    Ok(arr.into_any())
}
