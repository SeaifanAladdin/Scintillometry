##Give Credits Later

import numpy as np
from scipy import linalg
debug = False


SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"


def log(message):
    if debug:
        print str(message)
    

class Cholesky:
    def __init__(self, T):
        N = T.shape[0]
        self.L = np.zeros((N,N), complex)
        self.T = T

    def fact(self, method, p):
        T = self.T
        m = T.shape[1]
        n = T.shape[0]/m
        A1, A2 = self.__setup_gen(T, m)

        log(A1)
        log("")
        log(A2)
        self.L[:, :m] = A1
        log("")
        log("")
        log(self.L)

        for k in range(1,n):
            log("")
            log("k = " + str(k))
            ##Build generator at step k [A1(:e1, :) A2(s2:e2, :)]
            s1, e1, s2, e2 = self.__set_curr_gen(k, n, m)
            log("s1, e1, s2, e2 = {0}, {1}, {2}, {3}".format(s1,e1,s2,e2))
            log("")
            log("")
            if method==SEQ:
                A1, A2 = self.__seq_reduc(A1, A2, s1, e1, s2, e2, m)
            else:
                A1, A2 = self.__block_reduc(A1, A2, s1, e1, s2, e2, m, p, method)
            self.L[k*m:e2, k*m:(k + 1)*m]  = A1[:e1, :]
            log("new L at step k = \n{0}".format(self.L))
        return self.L

    ##Private Methods
    def __setup_gen(self, T, m):
        m = T.shape[1]
        n = T.shape[0]/m
        A1 = np.zeros(T.shape, complex)
        A2 = np.zeros(T.shape, complex)
        
        A1 = T.copy()
        
        c = linalg.cholesky(A1[:m,:m], lower=True)
        A1[:m, :m] = c
        A1[m:n*m,:] = A1[m:n*m,:].dot(linalg.inv(c[:m,:m].T))
        A2[m:n*m, :m] = 1j*A1[m:n*m, :m]
        return A1, A2

    def __set_curr_gen(self, k, n, m):
        s1 = 0
        e1 = (n - k)*m
        s2 = k*m
        e2 = n*m
        return s1, e1, s2, e2

    def __block_reduc(self, A1, A2, s1, e1, s2, e2, m, p, method):
        log("method = " + method)
        n = A1.shape[0]/m
        for sb1 in range (0, m, p):
            log("")
            sb2 = sb1 + s2
            eb1 = min(sb1 + p, m)
            eb2 = eb1 + s2
            u1 = eb1
            u2 = eb2
            p_eff = min(p, m - sb1)
            log("sb1, sb2, eb1, eb2, u1, u2, p_eff = {0}, {1}, {2}, {3}, {4}, {5}, {6}".format(sb1, sb2, eb1, eb2, u1, u2, p_eff))
            XX2 = np.zeros((p_eff, m), complex)
            for j in range(0, p_eff):
                log("")
                j1 = sb1 + j
                j2 = sb2 + j 
                log("j, j1, j2 = {0}, {1}, {2}".format(j, j1, j2))
                X2, beta, A1, A2 = self.__house_vec(A1, A2, j1, j2)
                XX2[j] = X2
                A1, A2 = self.__seq_update(A1, A2, X2, beta, eb1, eb2, j1, j2, m, n)
                S = self.__aggregate(XX2, beta, A2, p, j, j1, j2, p_eff, method)
            A1, A2 = self.__block_update(XX2, A1, sb1, eb1, u1, e1, A2, sb2, eb2, u2, e2, S, method)
            return A1, A2
    def __block_update(self,X2, A1, sb1, eb1, u1, e1, A2, sb2, eb2, u2, e2, S, method):
        def wy1():
            Y1, Y2 = S
            M = A1[u1:e1, sb1:eb1] + A2[u2:e2, :m].dot(X2[:p_eff, :m].T)
            log("M = " + str(M))
            A2[u2:e2, :m] = A2[u2:e2, :m] + M.dot(Y2[:m, :p_eff].T)
            A1[u1:e1, sb1:eb1] = A1[u1:e1, sb1:eb1] + M.dot(Y1.T[sb1:eb1, :p_eff])
            log("")
            log("Final A1 = " + str(A1))
            log("Final A2 = " + str(A2))
            return A1, A2
        def wy2():
            W1, W2 = S
            M = A1[u1:e1, sb1:eb1].dot(W1[sb1:eb1, :p_eff]) + A2[u2:e2, :m].dot(W2[:m, :p_eff])
            log("M = " + str(M))
            A1[u1:e1, sb1:eb1] = A1[u1:e1, sb1:eb1] + M
            A2[u2:e2, :m] = A2[u2:e2, :m] + M.dot(X2[:p_eff, :m])
            log("")
            log("Final A1 = " + str(A1))
            log("Final A2 = " + str(A2))
            return A1, A2
            
        log("")
        log("Block_update")
        m = A1.shape[1]
        n = A1.shape[0]/m
        nru = e1 - u1
        p_eff = eb1 - sb1 
        log("nru, p_eff = {}, {}".format(nru, p_eff))
        if method == WY1:
            return wy1()
        elif method == WY2:
            return wy2()

    def __aggregate(self, X2, beta, A2, p, j, j1, j2, p_eff, method):
        log("aggregate")
        
        def wy1():
            Y1 = np.zeros((m,p), complex)
            Y2 = np.zeros((m,p), complex)
            Y1[j1, j] = -beta
            Y2[:, j] =-beta*X2[j, :m]

            log("Y1_init = " + str(Y1))
            log("Y2_init = " + str(Y2))
            
            if (j > 0):
                v[: j - 1] = -beta*X2[j, :m].dot(Y2[:m, :j-1])
                Y1[j1, :j - 1] = Y1[j1, :j - 1] + v[:j - 1]
                Y2[:m, :j - 1] = Y2[:m, : j  - 1] + np.array([X2[j, :m]]).T.dot(v[:j - 1])
            log("")
            log("Y1_final = " + str(Y1))
            log("Y2_final = " + str(Y2))
            return Y1, Y2
        def wy2():
            W1 = np.zeros((m,p), complex)
            W2 = np.zeros((m, p), complex)
            W1[j1, j] = -beta
            W2[:,j] = -beta*X2[:, j]
            log("W1_init = " + str(W1))
            log("W2_init = " + str(W2))
            
            if j > 0:
                v[: j - 1] = -beta*X2[:j-1, :m].dot(X2.T[j, :m])
                W1[sb1:j1 - 1, j] = W1[sb1:j1 - 1, :j-1].dot(v[:j-1])
                W2[:m, j]= W2[:m, j] + W2[:m, :j-1].dot(v[:j-1])
            log("")
            log("W1_final = " + str(W1))
            log("W2_final = " + str(W2))
            return W1, W2
                
                
            
        m = A2.shape[1]
        n = A2.shape[0]/m
        sb1 = j1 - j
        sb2 = j2 - j
        v = np.zeros(m*(n + 1), complex) 
        log("sb1, sb2 = {0}, {1}".format(sb1, sb2)) 
        if method == WY1:
            return wy1()
        if method == WY2:
            return wy2()

    
    def __seq_reduc(self, A1, A2, s1, e1, s2, e2, m):
        n=  A1.shape[0]/m
        for j in range (0, m):
            j1 = j
            j2 = s2 + j
            log("")
            log("j, j1, j2 = {0}, {1}, {2}".format(j, j1, j2))

            X2, beta, A1, A2 = self.__house_vec(A1, A2, j1, j2)
            
            A1, A2 = self.__seq_update(A1, A2, X2, beta, e1, e2, j1, j2, m, n)
        return A1, A2

    def __seq_update(self, A1, A2, X2, beta, e1, e2, j1, j2, m, n):
        #X2 = np.array([X2])
        u1 = j1 + 1
        u2 = j2 + 1

        nru = e1 - u1
        log("u1, u2, nru = {0}, {1}, {2}".format(u1, u2, nru))

        
        v = np.zeros(m*(n + 1), complex)

        #log("A1[u1:e1, j1] = " + str(A1[u1:e1, j1])
        #log("A2[u2:e2, :] = " + str(A2[u2:e2, :])
        #log("X2 = " + str(X2)
        #log("X2.T = " + str(X2.T)

        v[:nru] = A1[u1:e1, j1] + A2[u2:e2, :].dot(X2.T)
        log("v = {0}".format(v[:nru]))
        A1[u1:e1, j1] = A1[u1:e1, j1] - beta*v[:nru]
        log("Final A1 = \n" + str(A1))

        if nru != 0:
            A2[u2:e2, :] = A2[u2:e2, :] - beta*np.array([v[:nru]]).T.dot(np.array([X2[:m]]))
        log("Final A2 = \n" + str(A2))
        return A1, A2

    def __house_vec(self, A1, A2, j1, j2):
        log("From house_vec:")
        X2 = np.zeros(A2[j2,:].shape, complex)
        if np.all(np.abs(A2[j2, :]) < 1e-13):
            beta = 0
        else:
            sigma = A2[j2, :].dot(A2[j2,:].T)
            alpha = (A1[j1,j1]**2 + sigma)**0.5
            log("sigma = " + str(sigma))
            log("alpha = " + str(alpha))
            if (np.abs(A1[j1,j1] + alpha) < np.abs(A1[j1, j1] - alpha)):
                z = A1[j1, j1]-alpha
                A1[j1,j1] = alpha
            else:
                z = A1[j1, j1]+alpha
                A1[j1,j1] = -alpha
            X2 = A2[j2,:]/z
            A2[j2, :] = A2[j2, :]/z
            
            beta = 2*z**2/(sigma + z**2)
        log("beta = " + str(beta))
        log("X2 = {0}".format(X2))
        log("")
        log("A1 = {0}".format(A1))
        log("\nA2 = {0}".format(A2))
        log("")
        return X2, beta, A1, A2
import interface