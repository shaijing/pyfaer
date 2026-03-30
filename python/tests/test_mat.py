from pyfaer._faer import FaerMatf64
import numpy as np
mat = FaerMatf64(2,3)
print(mat.shape)

mat1 = FaerMatf64.from_list([[1,2,3,4],[4,5,6,7]])

print(mat1.shape)

mat2 = FaerMatf64.from_list([[1,2,3,4],[4,5,6,7]])

mat3 = mat1 + mat2
print(mat3.to_numpy())

a = np.array([[1,2,3,4],[4,5,6,7]], dtype="float64")
mat4 = FaerMatf64.from_numpy(a)
print(mat4.to_numpy())

mat5 = FaerMatf64.from_numpy(a.T)


mat6 = mat4 @ mat5
print(mat6.to_numpy())