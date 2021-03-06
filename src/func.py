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


def createBlockedToeplitz(n,m, real=False):
    T_i = np.empty((n, m,m), complex)
    T = np.zeros((m*n, m*n), complex)
    T_c = []
    for i in range(n):
        T_temp = createToeplitz(m, real)
        T_c.append(T_temp[0][0])
        for j in range(n):
            if j + i + 1 <= n:
                T[(j)*m: (j+1)*m, (j + i)*m: (j+i+1)*m] = T_temp
            if j - i >=0:
                T[(j )*m: (j+1)*m, (j - i)*m: (j-i+1)*m] = np.conj(T_temp.T)
        T_i[i] = T_temp
    x0 = np.sum(np.abs(T_c))
    for i in range(len(T)):
        T[i, i] = x0
        
    
    return T


def createPaddedBlockedToeplitz(n, m, real=False):
    T = np.zeros((4*m*n, 4*m*n), complex)
    T_c = []
    for i in range(n):
        T_temp = createToeplitz(m, real)
        T2 = np.zeros((2*m, 2*m), complex)
        T2[:m,:m]= T_temp
        T2[m:, m:] = T_temp
        T_temp = T2
        T_c.append(T_temp[0][0])
        for j in range(2*n):
            if j + i + 1 <= 2*n:
                T[2*(j)*m: 2*(j+1)*m, 2*(j + i)*m: 2*(j+i+1)*m] = T_temp
            if j - i >= 0:
                T[2*(j )*m: 2*(j+1)*m, 2*(j - i)*m: 2*(j-i+1)*m] = np.conj(T_temp.T)
    
    #T[2*n*m:, 2*n*m:] = T[:2*n*m, :2*n*m]

    x0 = np.sum(np.abs(T_c))
    for i in range(len(T)):
        T[i, i] = x0
    return T
    

def LtoBlockedL(L, n, m):
    L_new = np.array(((n, m, m)),complex)
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

