import numpy as np
import pyfaer as pf
from pyfaer import FaerMat
import time

# 1. 准备数据 (6000x6000 f64)
size = 6000
a_l = [[float(i) for i in range(size)] for j in range(size)]

# --- PyFaer 准备 ---
a_pyf = FaerMat.from_list(a_l, dtype=pf.FaerDType.F64)
b_pyf = FaerMat.from_list(a_l, dtype=pf.FaerDType.F64)

# --- NumPy 准备 ---
a_np = np.array(a_l, dtype=np.float64)
b_np = np.array(a_l, dtype=np.float64)


def benchmark():
    # 2. 测试 PyFaer 性能
    start = time.perf_counter()
    for _ in range(5):  # 运行5次取平均
        c_pyf = a_pyf + b_pyf
    end = time.perf_counter()
    pyf_time = (end - start) / 5
    print(f"PyFaer 平均耗时: {pyf_time:.4f} 秒")

    # 3. 测试 NumPy 性能
    start = time.perf_counter()
    for _ in range(5):
        c_np = a_np + b_np
    end = time.perf_counter()
    np_time = (end - start) / 5
    print(f"NumPy 平均耗时:   {np_time:.4f} 秒")

    # 4. 计算加速比
    speedup = np_time / pyf_time
    print(f"---")
    print(f"加速比: {speedup:.2f}x (值越大表示 PyFaer 越快)")


if __name__ == "__main__":
    benchmark()
