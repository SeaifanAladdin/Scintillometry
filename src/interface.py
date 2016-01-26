from ToeplitzFactorizor import *
import numpy as np
from func import *
from sympy import *

np.random.seed(10)
N = int(raw_input('Choose the size of your matrix:\n'))
m = int(raw_input('Choose the size of your block:\n'))
T = generateHermetianT(N, m)
c = ToeplitzFactorizor(T)

print "Your generated Toeplitz Matrix:"
pprint(Matrix(T))
print 



method = raw_input('Choose a method among: seq wy1 wy2 yty1 yty2\n')

p = int(raw_input('Choose your p-factor (Not neccesary for seq):\n'))

L =  c.fact(method, p)

print "The cholesky decomposition of the generated Toeplitz Matrix:"
pprint(Matrix(np.around(L,2)))
