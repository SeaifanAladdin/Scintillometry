##Give Credits Later

import numpy as np
from scipy.linalg import inv, cholesky, triu
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + "/Exceptions")


from ToeplitzFactorizorExceptions import *

from mpi4py import MPI


from scipy import linalg


SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"
class ToeplitzFactorizor:
    
    def __init__(self, T):
        self.T = np.array(T, complex)
        self.n = size
        self.m = len(T)
        self.L = np.zeros((n*m, n*m), complex)

        

    def fact(self, method, p):
        if method not in np.array([SEQ, WY1, WY2, YTY1, YTY2]):
            raise InvalidMethodException(method)
        if p < 1 and method != SEQ:
            raise InvalidPException(p)
        
        m = self.m
        n = self.n
        A1, A2 = self.__setup_gen(self.T)

        self.L[rank*m:(rank + 1)*m,:m] = A1
            
        for k in range(1,n):
            ##Build generator at step k [A1(:e1, :) A2(s2:e2, :)]
            s1, e1, s2, e2 = self.__set_curr_gen(k, n)
            if method==SEQ:
                A1, A2 = self.__seq_reduc(A1, A2, s1, e1, s2, e2)
            else:
                A1, A2 = self.__block_reduc(A1, A2, s1, e1, s2, e2, m, p, method)
            if self.work1:
                self.L[(rank + k)*m:(rank + k + 1)*m, k*m:(k + 1)*m]  = A1
        L = self.L
        L_temp = np.array(comm.gather(L, root=0))
        if rank == 0:
            self.L = np.sum(L_temp, 0)
            return self.L

    ##Private Methods
    def __setup_gen(self, T):
        n = self.n
        A1 = np.zeros(T.shape, complex)
        A2 = np.zeros(T.shape, complex)
        cinv = None
        
        ##The root rank will compute the cholesky decomposition
        if rank == 0:
            c = cholesky(T, lower=True)
            cinv = inv(np.conj(c.T))
        cinv = comm.bcast(cinv, root=0)
        A1= T.dot(cinv)
        A2 = 1j*A1

        ##We are done with T. We shouldn't ever have a reason to use it again
        del self.T
        return A1, A2

    def __set_curr_gen(self, k, n):
        s1 = 0
        e1 = (n - k - 1)
        s2 = k
        e2 = n

        if rank <= e1:
            self.work1 = True
        else:
            self.work1 = False;
        if rank>= s2:
            self.work2 = True
        else:
            self.work2 = False
        return s1, e1, s2, e2

    def __block_reduc(self, A1, A2, s1, e1, s2, e2, m, p, method):
        
        n = A1.shape[0]/m
        M = np.zeros((m*n,m*n), dtype=complex)
        for sb1 in range (0, m, p):
            
            sb2 = sb1 + s2
            eb1 = min(sb1 + p, m)
            eb2 = eb1 + s2
            u1 = eb1
            u2 = eb2
            p_eff = min(p, m - sb1)
            
            XX2 = np.zeros((p_eff, m), complex)
            if method == WY1 or method == WY2:
                S = np.array([np.zeros((m,p)),np.zeros((m,p))], complex)
            elif method == YTY1 or YTY2:
                S = np.zeros((p, p), complex)
            for j in range(0, p_eff):
                
                j1 = sb1 + j
                j2 = sb2 + j 
                
                X2, beta, A1, A2 = self.__house_vec(A1, A2, j1, j2)
                XX2[j] = X2
                A1, A2 = self.__seq_update(A1, A2, X2, beta, eb1, eb2, j1, j2, m, n)
                S = self.__aggregate(S, XX2, beta, A2, p, j, j1, j2, p_eff, method)
            A1, A2 = self.__block_update(M, XX2, A1, sb1, eb1, u1, e1, A2, sb2, eb2, u2, e2, S, method)
        return A1, A2
    def __block_update(self,M, X2, A1, sb1, eb1, u1, e1, A2, sb2, eb2, u2, e2, S, method):
        def wy1():
            Y1, Y2 = S
            if nru == 0 or p_eff == 0: return A1, A2
            M[:nru,:p_eff] = A1[u1:e1, sb1:eb1] - A2[u2:e2, :m].dot(np.conj(X2)[:p_eff, :m].T)
            
            A2[u2:e2, :m] = A2[u2:e2, :m] + M[:nru,:p_eff].dot(Y2[:m, :p_eff].T)
            M[:nru,:p_eff] =  M[:nru,:p_eff].dot(Y1[sb1:eb1, :p_eff].T)
            A1[u1:e1, sb1:eb1] = A1[u1:e1, sb1:eb1] + M[:nru,:p_eff]
            
            
            
            return A1, A2
        def wy2():
            W1, W2 = S
            
            if nru == 0 or p_eff == 0: return A1, A2
            M[:nru,:p_eff] = A1[u1:e1, sb1:eb1].dot(W1[sb1:eb1, :p_eff]) - A2[u2:e2, :m].dot(np.conj(W2)[:m, :p_eff])
            
            A1[u1:e1, sb1:eb1] = A1[u1:e1, sb1:eb1] + M[:nru,:p_eff]
            A2[u2:e2, :m] = A2[u2:e2, :m] + M[:nru,:p_eff].dot(X2[:p_eff, :m])
            
            
            
            
            return A1, A2
        def yty1():
            T = S
            
            M[:nru,:p_eff] = A1[u1:e1, sb1:eb1] - A2[u2:e2, :m].dot(np.conj(X2)[:p_eff, :m].T)
            M[:nru,:p_eff] = M[:nru,:p_eff].dot(T[:p_eff, :p_eff])
            
            A1[u1:e1, sb1:eb1] = A1[u1:e1, sb1:eb1] + M[:nru,:p_eff]
            A2[u2:e2, :m] = A2[u2:e2, :m] + M[:nru, :p_eff].dot(X2[:p_eff, :m])
            
            
            
            return A1, A2

        def yty2():
            invT = S
            
            M[:nru,:p_eff] = A1[u1:e1, sb1:eb1] - A2[u2:e2, :m].dot(np.conj(X2)[:p_eff, :m].T)
            M[:nru,:p_eff] = M[:nru,:p_eff].dot(inv(invT[:p_eff, :p_eff]))
            
            A1[u1:e1, sb1:eb1] = A1[u1:e1, sb1:eb1] + M[:nru,:p_eff]
            A2[u2:e2, :m] = A2[u2:e2, :m] + M[:nru,:p_eff].dot(X2[:p_eff, :m])
            return A1, A2
        
        
        m = A1.shape[1]
        n = A1.shape[0]/m
        nru = e1 - u1
        p_eff = eb1 - sb1 
        
        if method == WY1:
            return wy1()
        elif method == WY2:
            return wy2()
        elif method ==YTY1:
            return yty1()
        elif method == YTY2:
            return yty2()

    def __aggregate(self,S,  X2, beta, A2, p, j, j1, j2, p_eff, method):
        
        
        def wy1():
            Y1 = S[0] ## it might be Y1 += new Y1
            Y2 = S[1]
            Y1[j1, j] = -beta
            Y2[:, j] =-beta*X2[j, :m]

            
            
            if (j > 0):
                v[: j ] = beta*np.conj(X2)[j, :m].dot(Y2[:m, :j])
                
                Y1[j1, :j] = Y1[j1, :j ] + v[:j ]
                Y2[:m, :j ] = Y2[:m, : j] + X2[j, :m][np.newaxis].T.dot(v[:j ][np.newaxis])
            
            
            
            return Y1, Y2
        def wy2():
            W1 = S[0]
            W2 = S[1]
            W1[j1, j] = -beta
            W2[:,j] = -beta*X2[j, :m]
            
            
            
            if j > 0:
                v[: j] = beta*X2[:j, :m].dot(np.conj(X2[j, :m].T))
                W1[sb1:j1, j] = W1[sb1:j1, :j].dot(v[:j])
                W2[:m, j]= W2[:m, j] + W2[:m, :j].dot(np.conj(v)[:j])
            
            
            return W1, W2
        def yty1():
            T = S
            T[j,j] = -beta
            if j > 0:
                v[:j] = beta*X2[:j, :m].dot(np.conj(X2)[j, :m].T)
                T[:j, j]=T[:j, :j].dot(v[:j])
            
            return T
        def yty2():
            invT = S
            
            if j == p_eff - 1:
                invT[:p_eff, :p_eff] = triu(X2[:p_eff, :m].dot(np.conj(X2)[:p_eff, :m].T))
                
                for jj in range(p_eff):
                    invT[jj,jj] = (invT[jj,jj] - 1.)/2.
            
            return invT
            
        m = A2.shape[1]
        n = A2.shape[0]/m
        sb1 = j1 - j
        sb2 = j2 - j
        v = np.zeros(m*(n + 1), complex) 
        
        if method == WY1:
            return wy1()
        if method == WY2:
            return wy2()
        if method == YTY1:
            return yty1()
        if method == YTY2:
            return yty2()

    
    def __seq_reduc(self, A1, A2, s1, e1, s2, e2):
        n = self.n
        for j in range (0, self.m):

            X2, beta, A1, A2 = self.__house_vec(A1, A2, j, s2)
            
            A1, A2 = self.__seq_update(A1, A2, X2, beta, e1, s2, j, m, n)
        return A1, A2

    def __seq_update(self, A1, A2, X2, beta, e1, s2, j, m, n):
        #X2 = np.array([X2])
        u = j + 1

        nru = e1*m - (s2*m + j + 1)
        if self.work2:
            B1 = A2.dot(np.conj(X2.T))
            if rank == s2:
                B1 = B1[u:]
            comm.Send(B1, dest=(rank - s2), tag=13)
        if self.work1:
           
            if rank == 0:
                B1 = np.empty(m - u, complex)
                comm.Recv(B1, source=(rank + s2), tag=13)
                B2 = A1[:, j]
                B2= B2[u:]
            else:
                B1 = np.empty(m, complex)
                comm.Recv(B1, source=(rank + s2), tag=13)
                B2 = A1[:, j]
                
            v = B2 - B1
            comm.Send(v, dest=(rank+s2), tag=14)
            if rank == 0:
                A1[u:,j] -= beta*v
            else:
                A1[:,j] -= beta*v
        if self.work2:
        
            if rank == s2:
                v = np.empty(m-u, complex)
                comm.Recv(v, source=(rank - s2), tag=14)
                A2[u:,:] -= beta*v[np.newaxis].T.dot(np.array([X2[:m]]))
            else:
                v = np.empty(m, complex)
                comm.Recv(v, source=(rank - s2), tag=14)
                A2 -= beta*v[np.newaxis].T.dot(np.array([X2[:m]]))
            
        return A1, A2

    def __house_vec(self, A1, A2, j, s2):
        isZero = False
        X2 = np.zeros(A2[j,:].shape, complex)
        beta = 0
        if rank==s2:
            if np.all(np.abs(A2[j, :]) < 1e-13):
                isZero=True
        isZero = comm.bcast(isZero, root=0)
        if isZero:
            return X2, beta, A1, A2
        
        if rank == s2:
            sigma = A2[j, :].dot(np.conj(A2[j,:]))
            comm.send(sigma, dest=0, tag=11)
        if rank == 0:
            sigma = comm.recv(source=s2, tag=11)
            alpha = (A1[j,j]**2 - sigma)**0.5            
            if (np.real(A1[j,j] + alpha) < np.real(A1[j, j] - alpha)):
                z = A1[j, j]-alpha
                A1[j,j] = alpha 
            else:
                z = A1[j, j]+alpha
                A1[j,j] = -alpha
            comm.send(z, dest=s2, tag=12)
            beta = 2*z*z/(-sigma + z*z)           
            
        if rank == s2:
            z = comm.recv(source=0, tag=12)
            X2 = A2[j,:]/z
            A2[j, :] = X2
        beta = comm.bcast(beta, root=0)
        X2 = comm.bcast(X2, root=s2) 

        return X2, beta, A1, A2


if __name__=="__main__":
    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()

    
    if len(sys.argv) != 4:
        print "error"
    else:
        from func import createBlockedToeplitz, testFactorization
        n = size
        m = int(sys.argv[2])
        method = sys.argv[1]
        p = int(sys.argv[3])
        T = None
        if rank == 0:
            T = createBlockedToeplitz(n, m)
        T1 = comm.bcast(T, root=0)[:m, rank*m:(rank+1)*m]
        c = ToeplitzFactorizor(T1)
        L = c.fact(method, p)
        
        if rank == 0 and not testFactorization(T, L):
            print "L error"
	
