// cast.rs

// 针对实数的通用转换 (利用泛型减少重复代码)
// fn cast_inner<T: RealField + Copy>(src: &FaerArray) -> Mat<T> {
//     match src {
//         // 利用 faer 的视图转换，这比手动 from_fn 更容易被编译器优化
//         FaerArray::F32(m) => m.as_ref().map(|&x| polymorphic_cast(x)).to_owned(),
//         FaerArray::F64(m) => m.as_ref().map(|&x| polymorphic_cast(x)).to_owned(),
//         // 复数转实数
//         FaerArray::C32(m) => m.as_ref().map(|&c| polymorphic_cast(c.re)).to_owned(),
//         FaerArray::C64(m) => m.as_ref().map(|&c| polymorphic_cast(c.re)).to_owned(),
//     }
// }

// /// 将 FaerArray 转换为目标 dtype
// pub fn cast_to(src: &FaerArray, target: FaerDType) -> FaerArray {
//     // 先把 src "解包"为 f64 或 c64 中间表示，再装入目标类型
//     // 利用一个 to_c64 和 to_f64 的中间层避免 O(n^2) 的分支
//     match target {
//         FaerDType::F32 => FaerArray::F32(cast_inner_f32(src)),
//         FaerDType::F64 => FaerArray::F64(cast_inner_f64(src)),
//         FaerDType::C32 => FaerArray::C32(cast_inner_c32(src)),
//         FaerDType::C64 => FaerArray::C64(cast_inner_c64(src)),
//     }
// }

// fn cast_inner_f32(src: &FaerArray) -> Mat<f32> {
//     match src {
//         FaerArray::F32(m) => m.clone(),
//         FaerArray::F64(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] as f32),
//         // 复数 → 实数：只取实部（与 numpy 行为一致，可酌情改为 panic）
//         FaerArray::C32(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)].re as f32),
//         FaerArray::C64(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)].re as f32),
//     }
// }

// // 以 cast_inner_f64 为例，其余同理
// fn cast_inner_f64(src: &FaerArray) -> Mat<f64> {
//     match src {
//         FaerArray::F32(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] as f64),
//         FaerArray::F64(m) => m.clone(),
//         // 复数 → 实数：只取实部（与 numpy 行为一致，可酌情改为 panic）
//         FaerArray::C32(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)].re as f64),
//         FaerArray::C64(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)].re),
//     }
// }

// fn cast_inner_c32(src: &FaerArray) -> Mat<Complex<f32>> {
//     match src {
//         FaerArray::F32(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| Complex {
//             re: m[(i, j)],
//             im: 0.0,
//         }),
//         FaerArray::F64(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| Complex {
//             re: m[(i, j)] as f32,
//             im: 0.0,
//         }),
//         FaerArray::C32(m) => m.clone(),
//         FaerArray::C64(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| Complex {
//             re: m[(i, j)].re as f32,
//             im: m[(i, j)].im as f32,
//         }),
//     }
// }

// fn cast_inner_c64(src: &FaerArray) -> Mat<Complex<f64>> {
//     match src {
//         FaerArray::F32(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| Complex {
//             re: m[(i, j)] as f64,
//             im: 0.0,
//         }),
//         FaerArray::F64(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| Complex {
//             re: m[(i, j)],
//             im: 0.0,
//         }),
//         FaerArray::C32(m) => Mat::from_fn(m.nrows(), m.ncols(), |i, j| Complex {
//             re: m[(i, j)].re as f64,
//             im: m[(i, j)].im as f64,
//         }),
//         FaerArray::C64(m) => m.clone(),
//     }
// }
