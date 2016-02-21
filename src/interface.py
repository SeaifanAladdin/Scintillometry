from ToeplitzFactorizor import *
import numpy as np
from func import *

np.random.seed(10)
N = int(raw_input('Choose the size of your matrix:\n'))
T = createToeplitz(N)
##print "Your Matrix is: "
##print((M))
##print
m = int(raw_input('Choose the size of your block:\n'))
c = ToeplitzFactorizor(T, m)

print "Your randomly generated toeplitz matrix consists of {0} {1}x{1} blocks".format(m, N/m)
print "Those blocks are..."
printBlocks(T)
print
print "Therefore, your toeplitz matrix is:"
print T
print


method = raw_input('Choose a method among: seq wy1 wy2 yty1 yty2\n')

p = int(raw_input('Choose your p-factor (Not neccesary for seq):\n'))

L =  c.fact(method, p)

print "The cholesky decomposition of the generated Toeplitz Matrix:"
print(np.around(L, 2))
