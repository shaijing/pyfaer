import numpy as np

# 精度轴移动
print(np.promote_types(np.int32, np.int64))  # -> int64

# 类别轴移动 (实数 -> 复数)
print(np.promote_types(np.float64, np.complex64))  # -> complex128

# 跨轴双重移动 (int64 + complex64)
# 1. 精度提升到 64位对齐
# 2. 类别提升到 Complex
print(np.promote_types(np.int64, np.complex64))  # -> complex128
