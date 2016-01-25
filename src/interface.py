import Cholesky as chol
import numpy as np
from func import *

np.random.seed(10)
N = int(raw_input('Choose the size of your matrix:\n'))
m = int(raw_input('Choose the size of your block:\n'))
T = generateHermetianT(N, m)
method = raw_input('Choose a method among: seq wy1 wy2 yty1 yty2\n')

p = -1
if method != chol.SEQ:
    p = int(raw_input('Choose your p-factor:\n'))
##method = "seq"



c = chol.Cholesky(T)
L =  c.fact(method, 2)

