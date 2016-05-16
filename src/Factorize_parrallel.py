##Give Credits Later

import numpy as np
from scipy.linalg import inv, cholesky, triu
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + "/Exceptions")


from ToeplitzFactorizorExceptions import *

from mpi4py import MPI


from scipy import linalg

from GeneratorBlocks import Blocks
from GeneratorBlock import Block


SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"
class ToeplitzFactorizor:
    
    def __init__(self, T, pad):
        self.comm = MPI.COMM_WORLD
        size  = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.T = np.array(T, complex)
        n = size
        self.n = n
        m = len(T)
        self.m = m
        self.pad = pad
        self.L = np.zeros((n*m*(1 + pad), n*m*(1 + pad)), complex)
        
        self.blocks = Blocks()
        
        self.numOfBlocks = n*(1 + pad)
        
        
        

        

    def fact(self, method, p):
        if method not in np.array([SEQ, WY1, WY2, YTY1, YTY2]):
            raise InvalidMethodException(method)
        if p < 1 and method != SEQ:
            raise InvalidPException(p)
        pad = self.pad
        m = self.m
        n = self.n
        self.__setup_gen(self.T)
        

        for b in self.blocks:
            self.L[b.rank*m:(b.rank + 1)*m,:m] = b.getA1()
        
            
        for k in range(1,n*(1 + pad)):
            ##Build generator at step k [A1(:e1, :) A2(s2:e2, :)]
            s1, e1, s2, e2 = self.__set_curr_gen(k, n)
            if method==SEQ:
                self.__seq_reduc(s1, e1, s2, e2)
                
            else:
                self.__block_reduc(s1, e1, s2, e2, m, p, method)
                
            for b in self.blocks:
                if b.rank <=e1:
                    self.L[(b.rank + k)*m:(b.rank + k + 1)*m, k*m:(k + 1)*m]  = b.getA1()

            
        L = self.L
        L_temp = np.array(self.comm.gather(L, root=0))
        
        if self.rank == 0:
            self.L = np.sum(L_temp, 0)
            return self.L

    ##Private Methods
    def __setup_gen(self, T):
        n = self.n
        m = self.m
        pad = self.pad
        A1 = np.zeros((m, m),complex)
        A2 = np.zeros((m, m), complex)
        cinv = None
        
        ##The root rank will compute the cholesky decomposition
        if self.rank == 0:
            c = cholesky(T, lower=False)
            cinv = inv(c)
        cinv = self.comm.bcast(cinv, root=0)
        A1= T.dot(cinv)
        b = Block(A1, self.rank)
        self.blocks.addBlock(b)
        if pad:
        	self.blocks.addBlock(Block(np.zeros((m,m), complex), self.rank + n))

        ##We are done with T. We shouldn't ever have a reason to use it again
        del self.T
        return A1, A2

    def __set_curr_gen(self, k, n):
        s1 = 0
        e1 = min(n, (n*(1 + self.pad) - k)) -1
        s2 = k
        e2 = e1 + s2
        for b in self.blocks:
            if s1 <= b.rank <=e1:
                b.setWork1(b.rank + k)
            else:
                b.setWork1(None)
            if e2 >= b.rank >= s2:
                b.setWork2(b.rank - k)
            else:
                b.setWork2(None)
        return s1, e1, s2, e2

    def __block_reduc(self, s1, e1, s2, e2, m, p, method):


        n = self.n
        M = np.zeros((m*n,m), dtype=complex)
        ch = 0
        for sb1 in range (0, m, p):
            for b in self.blocks:
                b.setWork(None, None)
                if b.rank==0: b.setWork1(s2)
                if b.rank==s2: b.setWork2(0)
        
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
                X2, beta= self.__house_vec(j1, s2) ##s2 or sb2?
                XX2[j] = X2

                self.__seq_update(X2, beta, eb1, eb2, s2, j1, m, n) ##is this good?
                S = self.__aggregate(S, XX2, beta, p, j, j1, j2, p_eff, method)

            self.__set_curr_gen(s2, n) ## Updates work
            self.__block_update(M, XX2, sb1, eb1, u1, e1, s2,  sb2, eb2, u2, e2, S, method)
            #raise Exception()
        return
    def __block_update(self,M, X2, sb1, eb1, u1, e1,s2, sb2, eb2, u2, e2, S, method):
        def wy1():
            Y1, Y2 = S
            if p_eff == 0: return
            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    s = u1
                A2 = b.getA2()
                B2 = A2[s:, :m].dot(np.conj(X2)[:p_eff,:m].T)    
                self.comm.Send(B2, dest=b.getWork2()%n, tag=3*num + b.getWork2())
                del A2

                    
            for b in self.blocks:
                if b.work1 == None: continue
                s = 0
                if b.rank == 0:
                    s=u1
                A1 = b.getA1()
                B1 = A1[s:, sb1:eb1]    
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=b.getWork1()%n, tag=3*num + b.rank)  
                M = B1 - B2
                self.comm.Send(M, dest=b.getWork1()%n, tag=4*num + b.rank)
                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M.dot(Y1[sb1:eb1, :p_eff].T) 
                del A1

            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    s = u1
                M = np.empty((m - s, p_eff), complex)
                self.comm.Recv(M, source=b.getWork2()%n, tag=4*num + b.getWork2())
                A2 = b.getA2()
                A2[s:, :m] = A2[s:,:m] + M.dot(Y2[:m, :p_eff].T)
                del A2
            
            return
           
        def wy2():
            W1, W2 = S
            if p_eff == 0: return
            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    s = u1
                A2 = b.getA2()
                B2 = A2[s:, :m].dot(np.conj(W2[:m,:p_eff])) 
                self.comm.Send(B2, dest=b.getWork2()%n, tag=3*num + b.getWork2())
                del A2

            for b in self.blocks:
                if b.work1 == None: continue
                s = 0
                if b.rank == 0:
                    s=u1
                A1 = b.getA1()
                B1 = B1 = A1[s:, sb1:eb1].dot(W1[sb1:eb1, :p_eff]) 
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=b.getWork1()%n, tag=3*num + b.rank)  
                M = B1 - B2
                self.comm.Send(M, dest=b.getWork1()%n, tag=4*num + b.rank)
                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M
                del A1          
   

            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    s = u1
                M = np.empty((m - s, p_eff), complex)
                self.comm.Recv(M, source=b.getWork2()%n, tag=4*num + b.getWork2())
                A2 = b.getA2()
                A2[s:, :m] = A2[s:,:m] + M.dot(X2)
                del A2 
            return 


        def yty1():
            T = S
            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    s = u1
                A2 = b.getA2()
                B2 = A2[s:, :m].dot(np.conj(X2[:p_eff, :m]).T)
                self.comm.Send(B2, dest=b.getWork2()%n, tag=3*num + b.getWork2())
                del A2
                
            for b in self.blocks:
                if b.work1 == None: continue
                s = 0
                if b.rank == 0:
                    s=u1
                A1 = b.getA1()
                B1 = A1[s:, sb1:eb1]
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=b.getWork1()%n, tag=3*num + b.rank)  
                M = B1 - B2
                M = M.dot(T[:p_eff,:p_eff])
                self.comm.Send(M, dest=b.getWork1()%n, tag=4*num + b.rank)
                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M
                del A1                  

            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    s = u1
                M = np.empty((m - s, p_eff), complex)
                self.comm.Recv(M, source=b.getWork2()%n, tag=4*num + b.getWork2())
                
                A2 = b.getA2()
                A2[s:, :m] = A2[s:,:m] + M.dot(X2)
                del A2 
            
            
            return

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
        
        
        m = self.m
        n = self.n
        nru = e1*m - u1
        p_eff = eb1 - sb1 
        num = self.numOfBlocks
        
        if method == WY1:
            return wy1()
        elif method == WY2:
            return wy2()
        elif method ==YTY1:
            return yty1()
        elif method == YTY2:
            return yty2()


    def __aggregate(self,S,  X2, beta, p, j, j1, j2, p_eff, method):
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
            
        m = self.m
        n = self.n
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



    def __seq_reduc(self, s1, e1, s2, e2):
        n = self.n
        m = self.m
        for j in range (0, self.m):
            X2, beta = self.__house_vec(j, s2)
            
            self.__seq_update(X2, beta, e1*m, e2*m, s2, j, m, n)

    def __seq_update(self,X2, beta, e1, e2, s2, j, m, n):
        #X2 = np.array([X2])
        u = j + 1
        num = self.numOfBlocks
        
        nru = e1*m - (s2*m + j + 1)
        for b in self.blocks:
            if b.work2 == None: 
                continue
            B1 = np.dot(b.getA2(), np.conj(X2.T))
            start = 0
            end = m
            if b.rank == s2:
                start = u
            if b.rank == e2/m:
                end = e2 % m or m
            B1 = B1[start:end]
            self.comm.Send(B1, dest=b.getWork2()%n, tag=4*num + b.getWork2())

        
        for b in self.blocks:
            if b.work1 == None: continue
            start = 0
            end = m
            if b.rank == 0:
                start = u
            if b.rank == e1/m:
                end = e1 % m or m

            B1 = np.empty(end-start, complex)
            
            self.comm.Recv(B1, source=b.getWork1()%n, tag=4*num + b.rank)
            A1 = b.getA1()
            B2 = A1[start:end, j]
                
            v = B2 - B1
            self.comm.Send(v, (b.getWork1())%n, 5*num + b.getWork1())
            A1[start:end,j] -= beta*v
            del A1

        for b in self.blocks:
            if b.work2 == None: 
                continue
            start = 0
            end = m
            if b.rank == s2:
                start = u
            if b.rank == e2/m :
                end = e2 % m or m
            v = np.empty(end-start,complex)
            self.comm.Recv(v, source=b.getWork2()%n, tag=5*num + b.rank)
            A2 = b.getA2()
            A2[start:end,:] -= beta*v[np.newaxis].T.dot(np.array([X2[:]]))
            del A2

    def __house_vec(self, j, s2):
        isZero = False
        X2 = np.zeros(self.m, complex)
        beta = 0
        blocks = self.blocks
        n = self.n
        num = self.numOfBlocks
        if blocks.hasRank(s2):
            A2 = blocks.getBlock(s2).getA2()
            if np.all(np.abs(A2[j, :]) < 1e-13):
                isZero=True
            del A2
        isZero = self.comm.bcast(isZero, root=s2%n)
        if isZero:
            return X2, beta
        
        if blocks.hasRank(s2):
            A2 = blocks.getBlock(s2).getA2()
            sigma = A2[j, :].dot(np.conj(A2[j,:]))
            self.comm.send(sigma, dest=0, tag=2*num + s2)
            del A2
        if blocks.hasRank(0):
            A1 = blocks.getBlock(0).getA1()
            sigma = self.comm.recv(source=s2%n, tag=2*num + s2)
            alpha = (A1[j,j]**2 - sigma)**0.5            
            if (np.real(A1[j,j] + alpha) < np.real(A1[j, j] - alpha)):
                z = A1[j, j]-alpha
                A1[j,j] = alpha 
            else:
                z = A1[j, j]+alpha
                A1[j,j] = -alpha
            self.comm.send(z, dest=s2%n, tag=3*num + s2)
            beta = 2*z*z/(-sigma + z*z)           
            del A1
            
        if blocks.hasRank(s2):
            z = self.comm.recv(source=0, tag=3*num + s2)
            A2 = blocks.getBlock(s2).getA2()
            X2 = A2[j,:]/z
            A2[j, :] = X2
            del A2
        beta = self.comm.bcast(beta, root=0)
        X2 = self.comm.bcast(X2, root=s2%n) 
        return X2, beta

if __name__=="__main__":
    np.random.seed(20)
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
        T1 = comm.bcast(T, root=0)[rank*m:(rank+1)*m, :m]
        c = ToeplitzFactorizor(T1)
        L = c.fact(method, p)
        if rank == 0:
            print testFactorization(T, L)
            import matplotlib.pyplot as plt
            plt.imshow(np.abs(T - L.dot(np.conj(L.T))))
            plt.colorbar()
            plt.show()
        if rank == 0 and not testFactorization(T, L):
            print np.around(L,1)
            print
            print np.around(cholesky(T, lower=True),1)
            print "L error"
	
