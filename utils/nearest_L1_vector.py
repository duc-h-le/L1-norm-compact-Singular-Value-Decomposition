for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from numpy.linalg import svd

sys.path.insert(1, '../lib')

from utils import *

def nearest_L1_vector (a, v):
    s_list = a/v
    L1_list = []
    for i in range(len(a)):
        L1_list = np.append(L1_list, sum(abs(a - s_list[i]*v)))
    arg = np.argmin(L1_list)
    s = s_list[arg]
    return s