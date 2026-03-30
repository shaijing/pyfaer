from textwrap import indent
from numpy.matlib import full
from pyfaer._faer import FaerMatf64
import numpy as np

zeros = FaerMatf64.zeros(2, 3)
print(zeros.to_numpy())
ones = FaerMatf64.ones(2, 3)
print(ones.to_numpy())

full = FaerMatf64.full(2, 3, 5.0)
print(full.to_numpy())

indentity = FaerMatf64.identity(4, 4)
print(indentity.to_numpy())
