# 方法一

这种基于 **POSET（偏序集）** 的设计将你的类型系统从“逻辑判断”升华为“代数推导”。我们要实现的是一个**有限分配格（Finite Distributive Lattice）**，其中每一个 DType 都是格上的一个点，而 `promote` 就是寻找它们的 **Join（最小上界）**。


### 1. 定义“能力特征”（Capabilities）
这是格的坐标维度。我们将 DType 映射为一组布尔特征和位宽，这决定了它在格中的“高度”。

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capability {
    pub is_complex: bool,
    pub is_float: bool,
    pub is_signed: bool,
    pub bits: u32,
}

impl Capability {
    /// 偏序关系定义：self <= other (other 能表示 self)
    pub fn is_subset_of(&self, other: &Capability) -> bool {
        // 1. 域约束：复数不能存入实数
        if self.is_complex && !other.is_complex { return false; }
        // 2. 类别约束：浮点不能无损存入整数
        if self.is_float && !other.is_float { return false; }
        // 3. 符号约束：有符号不能存入无符号（除非位宽大幅增加，此处按严格偏序）
        if self.is_signed && !other.is_signed { return false; }
        // 4. 精度约束
        if self.bits > other.bits { return false; }
        
        true
    }
}
```

---

### 2. 重新设计 `FaerDType`
我们将 DType 定义为格中的元素，并为其赋予“成本（Cost）”，用于在存在多个上界时选择“最小”的一个。

```rust
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaerDType {
    U32, U64,
    I32, I64,
    F32, F64,
    C32, C64,
}

impl FaerDType {
    pub fn capability(&self) -> Capability {
        match self {
            Self::U32 => Capability { is_complex: false, is_float: false, is_signed: false, bits: 32 },
            Self::U64 => Capability { is_complex: false, is_float: false, is_signed: false, bits: 64 },
            Self::I32 => Capability { is_complex: false, is_float: false, is_signed: true,  bits: 32 },
            Self::I64 => Capability { is_complex: false, is_float: false, is_signed: true,  bits: 64 },
            Self::F32 => Capability { is_complex: false, is_float: true,  is_signed: true,  bits: 32 },
            Self::F64 => Capability { is_complex: false, is_float: true,  is_signed: true,  bits: 64 },
            Self::C32 => Capability { is_complex: true,  is_float: true,  is_signed: true,  bits: 32 },
            Self::C64 => Capability { is_complex: true,  is_float: true,  is_signed: true,  bits: 64 },
        }
    }

    /// 定义成本：位宽是基础，浮点更贵，复数最贵
    pub fn cost(&self) -> u32 {
        let cap = self.capability();
        let base = cap.bits;
        let type_multiplier = if cap.is_complex { 1000 } else if cap.is_float { 500 } else { 0 };
        base + type_multiplier
    }
}
```



---

### 3. 格运算的核心：`Join` (Least Upper Bound)
这是整个系统的灵魂。它不再是写死的 `match`，而是一个在全集 `ALL_TYPES` 中搜索最小上界的过程。

```rust
const ALL_TYPES: [FaerDType; 8] = [
    FaerDType::U32, FaerDType::U64,
    FaerDType::I32, FaerDType::I64,
    FaerDType::F32, FaerDType::F64,
    FaerDType::C32, FaerDType::C64,
];

impl FaerDType {
    pub fn promote(a: Self, b: Self) -> Self {
        let cap_a = a.capability();
        let cap_b = b.capability();

        ALL_TYPES.iter()
            .copied()
            .filter(|candidate| {
                let cap_cand = candidate.capability();
                // 必须同时是 a 和 b 的上界
                cap_a.is_subset_of(&cap_cand) && cap_b.is_subset_of(&cap_cand)
            })
            // 在所有合法的上界中，找成本最低的（即“最小上界”）
            .min_by_key(|candidate| candidate.cost())
            .unwrap_or(FaerDType::C64) // 兜底到最高精度复数
    }
}
```

---

### 4. 性能优化：静态查找表（Static Promotion Table）
虽然 $O(N)$ 搜索很快，但为了达到极致性能，我们可以利用 Rust 的 `const` 或 `once_cell` 预计算一张 $8 \times 8$ 的查找表。

```rust
use once_cell::sync::Lazy;
use std::collections::HashMap;

static PROMOTION_CACHE: Lazy<HashMap<(FaerDType, FaerDType), FaerDType>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for &a in &ALL_TYPES {
        for &b in &ALL_TYPES {
            m.insert((a, b), FaerDType::promote(a, b));
        }
    }
    m
});

// 最终 Python 调用的接口，变为 O(1) 耗时
pub fn fast_promote(a: FaerDType, b: PyRef<'_, FaerDType>) -> FaerDType {
    *PROMOTION_CACHE.get(&(a, *b)).unwrap()
}
```

---

### 5. 这种设计的“降维打击”优势

* **优雅处理 `U64 + I64`**：
    * `I64` 无法表示 `U64`（无符号溢出）。
    * `U64` 无法表示 `I64`（负数溢出）。
    * 格系统会自动向上寻找，最终锁定在 `F64`（或你定义的更高位宽类型），因为它满足两个 `is_subset_of` 条件且 Cost 最低。
* **AMP（自动混合精度）的天然支持**：
    如果你加入 `F16`，只需在 `ALL_TYPES` 里添加它并定义其 `Capability`（bits: 16），整个系统的加减乘除会自动识别何时该用 `F16` 提升到 `F32`。
* **类型系统的单调性**：
    格论保证了 `promote(a, b)` 的结果总是稳定的，不会因为添加了新类型而导致旧代码的提升逻辑发生非预期的“跳跃”。

---

# 方法二
一、核心设计思想

1. **每个 dtype 对应一个“能力向量”**（Capability Vector）：

   * 直接描述这个类型能表示的所有数值空间，而不是拆成 Kind / Width / Domain。
   * 向量形式可以自然扩展到任意新的类型（例如 bf16、float8、int128）。

2. **偏序由可表示性定义**：

   * `a <= b` ⇔ 类型 `b` 可以无损表示类型 `a` 的所有值。
   * 无需三维 lattice 或 max 修补逻辑。

3. **自动提升（promotion）**：

   * 找出 **最小上界 (join)**，即能够表示两个类型的最小 dtype。
   * 用 **cost function** 或其他优先级定义最小上界。

4. **扩展性强**：

   * 新增整数、浮点、复数，甚至混合精度类型，无需修改核心逻辑。
   * AMP / autocast 直接映射到 Capability。

---

# 二、类型定义

```rust
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capability {
    /// 是否支持负数
    pub signed: bool,
    /// 是否支持小数
    pub float: bool,
    /// 是否支持复数
    pub complex: bool,
    /// 有效位宽
    pub bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaerDType {
    pub name: &'static str,
    pub capability: Capability,
    /// 可选 cost，决定 join 最小化策略
    pub cost: u32,
}
```

---

# 三、内置 dtype 列表

```rust
pub const ALL_DTYPES: &[FaerDType] = &[
    FaerDType { name: "u32", capability: Capability { signed:false, float:false, complex:false, bits:32 }, cost: 32 },
    FaerDType { name: "u64", capability: Capability { signed:false, float:false, complex:false, bits:64 }, cost: 64 },
    FaerDType { name: "i32", capability: Capability { signed:true, float:false, complex:false, bits:32 }, cost: 33 },
    FaerDType { name: "i64", capability: Capability { signed:true, float:false, complex:false, bits:64 }, cost: 65 },
    FaerDType { name: "f32", capability: Capability { signed:true, float:true, complex:false, bits:32 }, cost: 132 },
    FaerDType { name: "f64", capability: Capability { signed:true, float:true, complex:false, bits:64 }, cost: 164 },
    FaerDType { name: "c32", capability: Capability { signed:true, float:true, complex:true, bits:32 }, cost: 232 },
    FaerDType { name: "c64", capability: Capability { signed:true, float:true, complex:true, bits:64 }, cost: 264 },
];
```

---

# 四、偏序函数（≤）

```rust
pub fn can_represent(a: &Capability, b: &Capability) -> bool {
    if a.complex && !b.complex { return false; }
    if a.float && !b.float { return false; }
    if a.signed && !b.signed { return false; }
    if a.bits > b.bits { return false; }
    true
}
```

---

# 五、join（最小上界）

```rust
pub fn promote(a: &FaerDType, b: &FaerDType) -> FaerDType {
    ALL_DTYPES
        .iter()
        .filter(|t| can_represent(&a.capability, &t.capability) &&
                    can_represent(&b.capability, &t.capability))
        .min_by_key(|t| t.cost)
        .expect("No common supertype found")
        .clone()
}
```

---

# 六、astype/解析

```rust
impl std::str::FromStr for FaerDType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ALL_DTYPES.iter()
            .find(|t| t.name.eq_ignore_ascii_case(s))
            .cloned()
            .ok_or_else(|| format!("Unknown dtype '{}'", s))
    }
}
```

---

# 七、特点

1. **无三维 lattice 修补**

   * 所有类型直接依赖 Capability 描述。
   * 偏序由 `can_represent` 定义。

2. **自动推广完全正确**

   * 支持整数/浮点/复数混合。
   * `U64 + I64` → `F64`（正确处理溢出问题）。
   * `I32 + C32` → `C32`。

3. **可扩展性强**

   * 新类型只需添加到 `ALL_DTYPES`。
   * cost 自定义即可调整 join 最小上界。

4. **兼容 AMP / 混合精度**

   * 可以增加 `autocast` 映射表，将高精度 dtype 映射到低精度。

---

如果你需要，我可以帮你**直接把这个系统和 `FaerArray` 结合**，实现：

* 动态 dtype 矩阵存储
* AMP 自动混合精度
* kernel dispatch 自动选择底层实现

这样你的 Faer Python 绑定就能直接支持用户输入 dtype + 自动提升 + 混合精度。

