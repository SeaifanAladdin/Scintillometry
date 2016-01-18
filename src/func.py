import numpy as np


def generateM(N):
    A = np.abs(np.random.randn(N, N))
    M = A.dot(np.conj(A.T))
    return np.array(M, complex)

def createT(M, m):
    return M[:, :m]


def generateT(N, m):
    M = generateM(N)
    T = createT(M, m)
    return T
