from ToeplitzFactorizor import *
import numpy as np
from func import *

np.random.seed(10)
m = int(raw_input('Choose the size of your block:\n'))
n = int(raw_input('Choose the number of blocks:\n'))

T = createBlockedToeplitz(m,n)
print "Your Toeplit Matrix is: "
print(np.around(T,2))
print
c = ToeplitzFactorizor(T, m)



method = raw_input('Choose a method among: seq wy1 wy2 yty1 yty2\n')

p = int(raw_input('Choose your p-factor (Not neccesary for seq):\n'))

L =  c.fact(method, p)

print "The cholesky decomposition of the generated Toeplitz Matrix:"
print(np.around(L, 2))
