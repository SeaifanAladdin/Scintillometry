import numpy as np
from scipy import linalg

FORTRANFORMAT = "({}.d0, {}.d0)"
SIGMA = 1e-10



def toBlockedT(T, m):
    return T[:, :m]

def createToeplitz(N, real=False):
    x = np.random.random_integers(1, 10, N)
    x = np.array(x, dtype=complex)
    if not real:
        xj = np.random.random_integers(1, 10, N)
        x += 1j*xj
    x0 = np.sum(np.absolute(x))
    x[0] = x0
    T = linalg.toeplitz(x)
    return T


def generateBlockedT(N, m):
    T = createToeplitz(N)
    return toBlockedT(T, m)



def testFactorization(T, L):
    T_new = L.dot(np.conj(L.T))
    return np.all(np.abs(T - T_new) <= SIGMA)


def createTforFortran(T):
    m = T.shape[1]
    n = T.shape[0]/m
    s = ""
    s += "{}         {}".format(n,m)
    for j in range(0, n*m, m):
        for i in range(m):
            for k in range(j, j + m):
                s+="\n"
                s+=FORTRANFORMAT.format(np.array(np.real(T[k][i]), int), np.array(np.imag(T[k][i]), int))
            
    return s



def printBlocks(T):
    return
    m = T.shape[1]
    n = T.shape[0]/m
    
    for i in range(n):
        print "Block {}".format(i + 1)
        print(T[i*m:(i + 1)*m, :])

