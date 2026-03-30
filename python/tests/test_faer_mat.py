from pyfaer import FaerMat
import pyfaer as pf

a = FaerMat.from_list([[1, 2], [3, 4]])
print(a.shape)
print(a.dtype)
print(a)
