```rust

#[pyclass(eq, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Precision {
    I32,
    I64,
    F32,
    F64,
}

#[pyclass(eq, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Domain {
    Real,
    Complex,
}

#[pyclass(eq, skip_from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FaerDType {
    pub precision: Precision,
    pub domain: Domain,
}

  /// 利用格理论（Lattice）实现的类型提升
  #[staticmethod]
  pub fn promote(a: Self, b: Self) -> Self {
      use std::cmp::max;
      Self {
          precision: max(a.precision, b.precision),
          domain: max(a.domain, b.domain),
      }
  }
```
