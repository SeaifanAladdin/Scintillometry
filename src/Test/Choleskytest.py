##Written by Aladdin Seaifan (aladdin.seaifan@mail.utoronto.ca)
##This python file is to test the class CholeskyTest by using unittesting
import os
os.chdir("../")

import unittest
from Cholesky import *
import numpy as np
from func import generateM, createT, generateT



    

class CholeskyTest(unittest.TestCase):
    ##Testing initializing
    def test_initializingm1(self):
        for i in range(1, 10):
            self.initializeTest(i, 1)
            self.initializeTest(i, 2)

    def test_initializingm2(self):
        for i in range(2, 20, 2):
            self.initializeTest(i, 2)
            self.initializeTest(i, 2)

    def initializeTest(self, size, m):
        def initializeTTest(): 
            self.assertTrue((T == c.T).all())
            self.assertEqual(c.T.shape, T.shape)

        def initializeLTest():
            N = T.shape[0]
            L = np.zeros((N,N), complex)
            self.assertTrue((L == c.L).all())
            self.assertEqual(c.L.shape, L.shape)
            
        T = generateT(size, m)
        c = Cholesky(T)
        initializeTTest()
        initializeLTest()
    

            
    ##TODO: a test that would fail. I.E. shape = (N, m) where N % m != 0


    ##SEQ Test
    def test_seqm1N1(self):
        T = generateT(1,1)
        c = Cholesky(T)
        L = c.fact("seq", -1)
        self.LtoTTest(L, T)


    def test_seqm1varyingN(self):
        for N in range(1, 10):
            T = generateT(N, 1)
            c = Cholesky(T)
            L = c.fact("seq", -1)
            self.LtoTTest(L, T)
    def test_seqN20varyingm(self):
        N = 20
        M = generateT(N, N)
        for m in np.arange(1, N + 1):
           if N%m != 0: continue
           T = M[:, :m]
           c = Cholesky(T)
           L = c.fact("seq", -1)
           self.LtoTTest(L, T)

    ##WY1 Test
    def test_wy1_m1_N1_p1(self):
        T = generateT(1,1)
        c = Cholesky(T)
        L = c.fact("wy1", 1)
        self.LtoTTest(L, T)


    def test_wy1_m1_p1_varyingN(self):
        for N in range(1, 10):
            T = generateT(N, 1)
            c = Cholesky(T)
            L = c.fact("wy1", 1)
            self.LtoTTest(L, T)
            
    def test_wy1_N20_p1_varyingm(self):
        N = 20
        M = generateT(N, N)
        for m in np.arange(1, N + 1):
           if N%m != 0: continue
           T = M[:, :m]
           c = Cholesky(T)
           L = c.fact("wy1", 1)
           self.LtoTTest(L, T)

    def test_wy1_N20_m4_varying4(self):
        N = 20
        T = generateT(N, 4)
        for p in np.arange(1,10):
           print p
           c = Cholesky(T)
           L = c.fact("wy1", p)
           self.LtoTTest(L, T)


    



    

    def LtoTTest(self, L, T):
        M = L.dot(L.T)
        m = T.shape[1]
        N = T.shape[0]
        self.assertEqual(M.shape, (N,N))
        T_new = M[:, :m]
        self.assertTrue((np.abs(T - T_new) < 1e-13).all())
        
        
        
        


if __name__ == '__main__':
    unittest.main()    
