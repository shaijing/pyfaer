use faer::{Mat, MatMut, MatRef, complex::Complex, mat::AsMatMut};

use crate::{cast::CastResult, dtype::FaerDType};

pub enum FaerArray {
    F32(Mat<f32>),
    F64(Mat<f64>),
    C32(Mat<Complex<f32>>), // 复数 32 位 (float x 2)
    C64(Mat<Complex<f64>>), // 复数 64 位 (double x 2)
}

pub enum FaerArrayRef<'a> {
    F32(MatRef<'a, f32>),
    F64(MatRef<'a, f64>),
    C32(MatRef<'a, Complex<f32>>),
    C64(MatRef<'a, Complex<f64>>),
}
pub enum FaerArrayMut<'a> {
    F32(MatMut<'a, f32>),
    F64(MatMut<'a, f64>),
    C32(MatMut<'a, Complex<f32>>),
    C64(MatMut<'a, Complex<f64>>),
}
impl FaerArray {
    pub fn nrows(&self) -> usize {
        match self {
            Self::F32(m) => m.nrows(),
            Self::F64(m) => m.nrows(),
            Self::C32(m) => m.nrows(),
            Self::C64(m) => m.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::F32(m) => m.ncols(),
            Self::F64(m) => m.ncols(),
            Self::C32(m) => m.ncols(),
            Self::C64(m) => m.ncols(),
        }
    }
    pub fn dtype(&self) -> FaerDType {
        match self {
            Self::F32(_) => FaerDType::F32,
            Self::F64(_) => FaerDType::F64,
            Self::C32(_) => FaerDType::C32,
            Self::C64(_) => FaerDType::C64,
        }
    }
    pub fn as_ref(&self) -> FaerArrayRef<'_> {
        match self {
            Self::F32(m) => FaerArrayRef::F32(m.as_ref()),
            Self::F64(m) => FaerArrayRef::F64(m.as_ref()),
            Self::C32(m) => FaerArrayRef::C32(m.as_ref()),
            Self::C64(m) => FaerArrayRef::C64(m.as_ref()),
        }
    }
    pub fn as_mut(&mut self) -> FaerArrayMut<'_> {
        match self {
            Self::F32(m) => FaerArrayMut::F32(m.as_mat_mut()),
            Self::F64(m) => FaerArrayMut::F64(m.as_mut()),
            Self::C32(m) => FaerArrayMut::C32(m.as_mut()),
            Self::C64(m) => FaerArrayMut::C64(m.as_mut()),
        }
    }
    pub fn cast_to(&self, target: FaerDType) -> CastResult<'_> {
        // 同类型：直接取 MatRef，零拷贝
        let same = matches!(
            (self, target),
            (Self::F32(_), FaerDType::F32)
                | (Self::F64(_), FaerDType::F64)
                | (Self::C32(_), FaerDType::C32)
                | (Self::C64(_), FaerDType::C64)
        );
        if same {
            return CastResult::Borrowed(self.as_ref());
        }

        // 跨类型：分配新 Mat
        CastResult::Owned(match (self, target) {
            (Self::F32(m), FaerDType::F64) => {
                FaerArray::F64(Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] as f64))
            }
            (Self::F32(m), FaerDType::C32) => {
                FaerArray::C32(Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
                    Complex::new(m[(i, j)], 0.0)
                }))
            }
            (Self::F32(m), FaerDType::C64) => {
                FaerArray::C64(Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
                    Complex::new(m[(i, j)] as f64, 0.0)
                }))
            }
            (Self::F64(m), FaerDType::C64) => {
                FaerArray::C64(Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
                    Complex::new(m[(i, j)], 0.0)
                }))
            }
            (Self::C32(m), FaerDType::C64) => {
                FaerArray::C64(Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
                    Complex::new(m[(i, j)].re as f64, m[(i, j)].im as f64)
                }))
            }
            (src, tgt) => panic!("不支持的转换：从 {:?} 到 {:?}", src.dtype(), tgt),
        })
    }
}
