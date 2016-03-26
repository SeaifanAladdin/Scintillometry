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
        self.comm = MPI.COMM_WORLD
        size  = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.T = np.array(T, complex)
        self.npn = T.shape[0]/T.shape[1]
        n = self.npn*size
        
        self.n = n
        m = T.shape[1]
        self.m = m
        self.L = np.zeros((n*m, n*m), complex)

        

    def fact(self, method, p):
        if method not in np.array([SEQ, WY1, WY2, YTY1, YTY2]):
            raise InvalidMethodException(method)
        if p < 1 and method != SEQ:
            raise InvalidPException(p)
        
        m = self.m
        n = self.n
        npn = self.npn
        A1, A2 = self.__setup_gen(self.T)



        self.L[npn*self.rank*m:npn*(self.rank + 1)*m,:m] = A1
            
        for k in range(1,n):
            ##Build generator at step k [A1(:e1, :) A2(s2:e2, :)]
            s1, e1, s2, e2 = self.__set_curr_gen(k, n)
            if method==SEQ:
                A1, A2 = self.__seq_reduc(A1, A2, s1, e1, s2, e2)
                
            else:
                A1, A2 = self.__block_reduc(A1, A2, s1, e1, s2, e2, m, p, method)
            if self.rank <= e1:
                self.L[(self.rank + k)*m:(self.rank + k + 1)*m, k*m:(k + 1)*m]  = A1
        L = self.L
        L_temp = np.array(self.comm.gather(L, root=0))
        
        if self.rank == 0:
            self.L = np.sum(L_temp, 0)
            return self.L

    ##Private Methods
    def __setup_gen(self, T):
        n = self.n
        m = self.m
        A1 = np.zeros(T.shape, complex)
        A2 = np.zeros(T.shape, complex)
        cinv = None
        
        ##The root rank will compute the cholesky decomposition
        if self.rank == 0:
            print T[:m,:m].shape; c = cholesky(T[:m,:m], lower=True)
            cinv = inv(np.conj(c.T))
        cinv = self.comm.bcast(cinv, root=0)
        A1= T[:,:].dot(cinv)
        A2 = 1j*A1

        ##We are done with T. We shouldn't ever have a reason to use it again
        del self.T
        return A1, A2

    def __set_curr_gen(self, k, n):
        npn = self.npn
        s1 = 0
        e1 = (n - k - 1)
        s2 = k
        e2 = n
        if self.rank <= e1/npn:
            self.work1 = e1 %npn or npn
            print "work1", self.work1
        else:
            self.work1 = False;
        if self.rank>= s2/npn:
            self.work2 = s2 %npn or npn
            print "work2", self.work2
        else:
            self.work2 = False
        return s1, e1, s2, e2

    def __block_reduc(self, A1, A2, s1, e1, s2, e2, m, p, method):
        self.work1 = False
        self.work2 = False
        if self.rank == 0:
            self.work1 = True
        if self.rank == s2:
            self.work2 = True

        n = A1.shape[0]/m
        M = np.zeros((m*n,m), dtype=complex)
        ch = 0
        for sb1 in range (0, m, p):
            sb2 = s2*m + sb1
            eb1 = min(sb1 + p, m) #next j
            eb2 = s2*m + eb1
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
                X2, beta, A1, A2 = self.__house_vec(A1, A2, j1, s2) ##s2 or sb2?
                XX2[j] = X2

                A1, A2 = self.__seq_update(A1, A2, X2, beta, eb1, eb2, s2, j1, m, n) ##is this good?
                S = self.__aggregate(S, XX2, beta, A2, p, j, j1, j2, p_eff, method)

                
            A1, A2 = self.__block_update(M, XX2, A1, sb1, eb1, u1, e1, s2, A2, sb2, eb2, u2, e2, S, method)
            #raise Exception()
        return A1, A2
    def __block_update(self,M, X2, A1, sb1, eb1, u1, e1,s2, A2, sb2, eb2, u2, e2, S, method):
        def wy1():
            Y1, Y2 = S
            if p_eff == 0: return A1, A2
            if self.rank >= s2:
                    s = 0
                    if self.rank == s2:
                        s = u1
                    B2 = A2[s:, :m].dot(np.conj(X2)[:p_eff,:m].T)
                    self.comm.Send(B2, dest=(self.rank- s2), tag=15)
                    M = np.empty((m - s, p_eff), complex)
                    
                    self.comm.Recv(M, source=(self.rank - s2), tag=16)


                    A2[s:, :m] = A2[s:,:m] + M.dot(Y2[:m, :p_eff].T)
                   
            if self.rank<=e1:
               
                    s = 0
                    if self.rank == 0:
                        s = u1
                    B1 = A1[s:, sb1:eb1]
                    
                    B2 = np.empty((m - s, p_eff), complex)
                    self.comm.Recv(B2, source=(self.rank + s2), tag=15)
                    M = B1 - B2
                    self.comm.Send(M, dest=(self.rank + s2), tag=16)
                    A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M.dot(Y1[sb1:eb1, :p_eff].T)
            
            return A1, A2
        def wy2():
            W1, W2 = S
            if p_eff == 0: return A1, A2
            if self.rank >= s2:
                    s = 0
                    if self.rank == s2:
                        s = u1
                    B2 = A2[s:, :m].dot(np.conj(W2[:m,:p_eff]))
                    self.comm.Send(B2, dest=(self.rank- s2), tag=15)
                    M = np.empty((m - s, p_eff), complex)
                    
                    self.comm.Recv(M, source=(self.rank - s2), tag=16)


                    A2[s:, :m] = A2[s:,:m] + M.dot(X2)
            if self.rank<=e1:
                s = 0
                if self.rank == 0:
                    s = u1
                B1 = A1[s:, sb1:eb1].dot(W1[sb1:eb1, :p_eff])
                
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=(self.rank + s2), tag=15)
                M = B1 - B2
                self.comm.Send(M, dest=(self.rank + s2), tag=16)
                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M

            return A1, A2     


        def yty1():
            T = S
            if self.rank >= s2:
                s = 0
                if self.rank == s2:
                    s = u1
                B2 = A2[s:, :m].dot(np.conj(X2[:p_eff, :m]).T)
                self.comm.Send(B2, dest=(self.rank- s2), tag=15)
                M = np.empty((m - s, p_eff), complex)
                
                self.comm.Recv(M, source=(self.rank - s2), tag=16)
                A2[s:, :m] = A2[s:,:m] + M.dot(X2)
            if self.rank<=e1:
                s = 0
                if self.rank == 0:
                    s = u1
                B1 = A1[s:, sb1:eb1]
                
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=(self.rank + s2), tag=15)
                M = B1 - B2
                M = M.dot(T[:p_eff,:p_eff])
                self.comm.Send(M, dest=(self.rank + s2), tag=16)

                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M

            
            
            return A1, A2

        def yty2():
            invT = S

            if self.rank >= s2:
                s = 0
                if self.rank == s2:
                    s = u1
                B2 = A2[s:, :m].dot(np.conj(X2[:p_eff, :m]).T)
                self.comm.Send(B2, dest=(self.rank- s2), tag=15)
                M = np.empty((m - s, p_eff), complex)
                
                self.comm.Recv(M, source=(self.rank - s2), tag=16)
                A2[s:, :m] = A2[s:,:m] + M.dot(X2)
            if self.rank<=e1:
                s = 0
                if self.rank == 0:
                    s = u1
                B1 = A1[s:, sb1:eb1]
                
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=(self.rank + s2), tag=15)
                M = B1 - B2
                M = M.dot(inv(invT[:p_eff,:p_eff]))
                self.comm.Send(M, dest=(self.rank + s2), tag=16)

                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M
            
            return A1, A2
        
        
        m = A1.shape[1]
        n = A1.shape[0]/m
        nru = e1*m - u1
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
        #log("aggregate")
        
        def wy1():
            Y1 = S[0] ## it might be Y1 += new Y1
            Y2 = S[1]
            Y1[j1, j] = -beta
            Y2[:, j] =-beta*X2[j, :m]

            #log("Y1_init = " + str(Y1))
            #log("Y2_init = " + str(Y2))
            if (j > 0):
                v[: j ] = beta*np.conj(X2)[j, :m].dot(Y2[:m, :j])
                #log("v = {}".format(v))
                Y1[j1, :j] = Y1[j1, :j ] + v[:j ]
                Y2[:m, :j ] = Y2[:m, : j] + X2[j, :m][np.newaxis].T.dot(v[:j ][np.newaxis])
            #log("")
            #log("Y1_final = " + str(Y1))
            #log("Y2_final = " + str(Y2))
            return Y1, Y2
        def wy2():
            W1 = S[0]
            W2 = S[1]
            W1[j1, j] = -beta
            W2[:,j] = -beta*X2[j, :m]
            #log("W1_init = " + str(W1))
            #log("W2_init = " + str(W2))
            
            if j > 0:
                v[: j] = beta*X2[:j, :m].dot(np.conj(X2[j, :m].T))
                W1[sb1:j1, j] = W1[sb1:j1, :j].dot(v[:j])
                W2[:m, j]= W2[:m, j] + W2[:m, :j].dot(np.conj(v)[:j])
            #log("")
            #log("W1_final = " + str(W1))
            #log("W2_final = " + str(W2))
            return W1, W2
        def yty1():
            T = S
            T[j,j] = -beta
            if j > 0:
                v[:j] = beta*X2[:j, :m].dot(np.conj(X2)[j, :m].T)
                T[:j, j]=T[:j, :j].dot(v[:j])
            #log("T = " + str(T))
            return T
        def yty2():
            invT = S
            #log("old invT = " + str(invT))
            if j == p_eff - 1:
                invT[:p_eff, :p_eff] = triu(X2[:p_eff, :m].dot(np.conj(X2)[:p_eff, :m].T))
                #log("invT = " + str(invT))
                for jj in range(p_eff):
                    invT[jj,jj] = (invT[jj,jj] - 1.)/2.
            #log("invT = {}".format(invT))
            return invT
            
        m = A2.shape[1]
        n = A2.shape[0]/m
        sb1 = j1 - j
        sb2 = j2 - j
        v = np.zeros(m*(n + 1), complex) 
        #log("sb1, sb2 = {0}, {1}".format(sb1, sb2)) 
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
        m = self.m
        for j in range (0, self.m):
            X2, beta, A1, A2 = self.__house_vec(A1, A2, j, s2)
            
            A1, A2 = self.__seq_update(A1, A2, X2, beta, e1*m, e2*m, s2, j, m, n)
        return A1, A2

    def __seq_update(self, A1, A2, X2, beta, e1, e2, s2, j, m, n):
        #X2 = np.array([X2])
        npn = self.npn
        offset = (s2 % npn)*m

        j1 = j+ offset
        u = j + 1
        u1 = j1 + 1
        npn = self.npn

        nru = e1*m - (s2*m + j + 1)
        if self.work2:
            B1 = A2.dot(np.conj(X2.T))
            start = 0
            end = npn*m
            if self.rank == s2/npn:
                start = u1
            if self.rank == e2/(npn*m):
                end = e2 % npn*m or npn*m
            B1 = B1[start:end]
            
            self.comm.Send(B1, dest=(self.rank - s2/npn), tag=13)
        
        if self.work1:
            start = 0
            end = npn*m
            if self.rank == 0:
                start = u
            if self.rank == e1/(m*npn):
                end = e1 % npn*m or npn*m
            B1 = np.empty(end-start, complex)
            print (self.rank - s2/npn)
            self.comm.Recv(B1, source=(self.rank + s2/npn), tag=13)
            B2 = A1[start:end, j]
                
            v = B2 - B1
            self.comm.Send(v, dest=(self.rank+s2/npn), tag=14)
            A1[start:end,j] -= beta*v
        

        
        if self.work2:
            start = 0
            end = npn*m
            if self.rank == s2/npn:
                start = u1
            if self.rank == e2/(npn*m) :
                end = e2 % npn*m or npn*m
            v = np.empty(end-start,complex)
            self.comm.Recv(v, source=(self.rank - s2/npn), tag=14)
            A2[start:end,:] -= beta*v[np.newaxis].T.dot(np.array([X2[:]]))

        return A1, A2

    def __house_vec(self, A1, A2, j, s2):
        npn = self.npn
        offset = (s2 % npn)*m

        j1 = j+ offset

        isZero = False
        X2 = np.zeros(A2[j1,:m].shape, complex)
        beta = 0
        if self.rank==s2/npn:
            if np.all(np.abs(A2[j1, :m]) < 1e-13):
                isZero=True
        isZero = self.comm.bcast(isZero, root=0)
        if isZero:
            return X2, beta, A1, A2
        
        if self.rank == s2/npn:
            sigma = A2[j1, :m].dot(np.conj(A2[j1,:m]))
            self.comm.send(sigma, dest=0, tag=11)
        if self.rank == 0:
            sigma = self.comm.recv(source=s2/npn, tag=11)
            alpha = (A1[j1,j]**2 - sigma)**0.5            
            if (np.real(A1[j1,j] + alpha) < np.real(A1[j1, j] - alpha)):
                z = A1[j1, j]-alpha
                A1[j1,j] = alpha 
            else:
                z = A1[j1, j]+alpha
                A1[j1,j] = -alpha
            self.comm.send(z, dest=s2/npn, tag=12)
            beta = 2*z*z/(-sigma + z*z)           
            
        if self.rank == s2/npn:
            z = self.comm.recv(source=0, tag=12)
            X2 = A2[j1,:]/z
            A2[j1, :] = X2
        beta = self.comm.bcast(beta, root=0)
        X2 = self.comm.bcast(X2, root=s2/npn) 

        return X2, beta, A1, A2

if __name__=="__main__":
    np.random.seed(20)
    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()

    
    if len(sys.argv) != 5:
        print "error"
    else:
        from func import createBlockedToeplitz, testFactorization
        n = int(sys.argv[2])
        m = int(sys.argv[3])
        method = sys.argv[1]
        p = int(sys.argv[4])
        T = None
        
        npn = n/size
        if rank == 0:
            T = createBlockedToeplitz(n, m)
        T1 = comm.bcast(T, root=0)[npn*rank*m:npn*(rank+1)*m, :m]
        c = ToeplitzFactorizor(T1)
        L = c.fact(method, p)
        if rank == 0:
            print testFactorization(T, L)
        if rank == 0 and not testFactorization(T, L):
            print np.around(L,1)
            print
            print np.around(cholesky(T, lower=True),1)
            print "L error"
	
