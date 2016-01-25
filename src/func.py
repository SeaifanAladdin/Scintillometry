import numpy as np

FORTRANFORMAT = "({}.d0, {}.d0)"

def generateA(N):
    A = np.abs(np.random.random_integers(1, 10, (N, N)))
    return np.array(A, complex)

def generateM(N):
    A = generateA(N)
    M = A.dot(np.conj(A.T))
    return M

def generateHermetianM(N):
    A = generateA(N) + 1j*generateA(N)
    M = A.dot(np.conj(A.T))
    return M

def createT(M, m):
    if len(M) != len(M[0]):
        raise Exception()
    return M[:, :m]


def generateT(N, m):
    M = generateM(N)
    T = createT(M, m)
    return T

def generateHermetianT(N, m):
    M = generateHermetianM(N)
    T = createT(M, m)
    return T


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

