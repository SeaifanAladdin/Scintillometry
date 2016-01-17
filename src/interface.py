import Cholesky as chol
import numpy as np

M = np.abs(np.random.randn(8, 8))
M = np.array(M, complex)
M += np.random.randn(8,8)*1j
M = M.dot(np.conj(M.T))
#M = np.array([[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]], complex)
#M = np.array([[2,1],[1,2]], complex)
T = M[:, :2]
#T = np.array([[ 1.12767849], [ 0.47295382],[ 2.04640699],[ 0.98543626]])

#method = raw_input('Choose a method among: seq wy1 wy2 yty1 yty2')
method = "wy1"

c = chol.Cholesky(T)
L =  c.fact(method, 2)

print np.real(L.dot(np.conj(L.T)) - M) < 1e-10

