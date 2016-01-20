import Cholesky as chol
import numpy as np
from func import *

np.random.seed(10)
m = 5
T = generateT(15, m)

##method = raw_input('Choose a method among: seq wy1 wy2 yty1 yty2')
method = "yty2"



c = chol.Cholesky(T)
L =  c.fact(method, 2)

print np.real(L.dot(np.conj(L.T))[:, :m] - T) < 1e-10


