import numpy as np
import pyfaer as pf
from pyfaer import FaerMat
import time


def benchmark(a_pyf, b_pyf, a_np, b_np):
    # 1. 测试 PyFaer 原地加法 (+=)
    start = time.perf_counter()
    for _ in range(5):
        a_pyf += b_pyf  # 确保你的 Rust 绑定实现了 __iadd__
    end = time.perf_counter()
    pyf_time = (end - start) / 5
    print(f"PyFaer 平均耗时: {pyf_time:.4f} 秒")

    # 2. 测试 NumPy 原地加法 (+=)
    start = time.perf_counter()
    for _ in range(5):
        a_np += b_np  # NumPy 的 += 是原地操作，不产生新内存分配
    end = time.perf_counter()
    np_time = (end - start) / 5
    print(f"NumPy 平均耗时:   {np_time:.4f} 秒")

    print(f"---")
    print(f"加速比: {np_time / pyf_time:.2f}x")
    a_pyf_final = a_pyf.to_numpy()  # 假设有这个方法

    if np.allclose(a_pyf_final, a_np):
        print("✅ 结果校验通过：PyFaer 和 NumPy 计算结果一致")
    else:
        print("❌ 结果校验失败：计算结果不匹配！")


if __name__ == "__main__":
    size = 6
    a_l = [[float(i) for i in range(size)] for j in range(size)]

    # 准备数据
    a_pyf = FaerMat.from_list(a_l)
    b_pyf = FaerMat.from_list(a_l)
    a_np = np.array(a_l, dtype=np.float64)
    b_np = np.array(a_l, dtype=np.float64)

    benchmark(a_pyf, b_pyf, a_np, b_np)
