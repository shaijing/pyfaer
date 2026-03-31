use faer::{Mat, complex::Complex};

pub enum FaerArray {
    F32(Mat<f32>),
    F64(Mat<f64>),
    C32(Mat<Complex<f32>>), // 复数 32 位 (float x 2)
    C64(Mat<Complex<f64>>), // 复数 64 位 (double x 2)
}
