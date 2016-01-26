##Written by Aladdin Seaifan (aladdin.seaifan@mail.utoronto.ca)
##This python file is to test the class CholeskyTest by using unittesting
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0, parentdir + "/Exceptions")


import unittest
from Cholesky import *
import numpy as np
from func import *

class CholeskyTest(unittest.TestCase):
    def setUp(self):
        pass

    ## Initializing Test
    ##TODO, Fail when T isn't a valid toeplitz matrix
    ##TODO, Fails when T[:,m] isn't hermetian

    ## Seq Test
    def test_seq_real_size1(self):
        
        c = self.__setupC(1,1, True)
        self.__methodSuccessfulTest("seq", c, -1)

    def test_seq_real_N20m4(self):
        c = self.__setupC(20,4, True)
        self.__methodSuccessfulTest("seq", c, -1)

    def test_seq_real_N20m20(self):
        c = self.__setupC(20,20, True)
        self.__methodSuccessfulTest("seq", c, -1)

    def test_seq_complex_size1(self):
        
        c = self.__setupC(1,1, False)
        self.__methodSuccessfulTest("seq", c, -1)

    def test_seq_complex_N20m4(self):
        c = self.__setupC(20,4, False)
        self.__methodSuccessfulTest("seq", c, -1)

    def test_seq_complex_N20m20(self):
        c = self.__setupC(20,20, False)
        self.__methodSuccessfulTest("seq", c, -1)


    ## wy1 Test
    def test_wy1_real_size1(self):
        
        c = self.__setupC(1,1, True)
        self.__methodSuccessfulTest("wy1", c, 1)

    def test_wy1_real_N20m4(self):
        c = self.__setupC(20,4, True)
        self.__methodSuccessfulTest("wy1", c, 2)

    def test_wy1_real_N20m20(self):
        c = self.__setupC(20,20, True)
        self.__methodSuccessfulTest("wy1", c, 5)

    def test_wy1_complex_size1(self):
        
        c = self.__setupC(1,1, False)
        self.__methodSuccessfulTest("wy1", c, 1)

    def test_wy1_complex_N20m4p2(self):
        c = self.__setupC(20,4, False)
        self.__methodSuccessfulTest("wy1", c, 2)

    def test_wy1_complex_N20m20p5(self):
        c = self.__setupC(20,20, False)
        self.__methodSuccessfulTest("wy1", c, 5)



    ## wy2 Test
    def test_wy2_real_size1(self):
        
        c = self.__setupC(1,1, True)
        self.__methodSuccessfulTest("wy2", c, 1)

    def test_wy2_real_N20m4(self):
        c = self.__setupC(20,4, True)
        self.__methodSuccessfulTest("wy2", c, 2)

    def test_wy2_real_N20m20(self):
        c = self.__setupC(20,20, True)
        self.__methodSuccessfulTest("wy2", c, 5)

    def test_wy2_complex_size1(self):
        
        c = self.__setupC(1,1, False)
        self.__methodSuccessfulTest("wy2", c, 1)

    def test_wy2_complex_N20m4p2(self):
        c = self.__setupC(20,4, False)
        self.__methodSuccessfulTest("wy2", c, 2)

    def test_wy2_complex_N20m20p5(self):
        c = self.__setupC(20,20, False)
        self.__methodSuccessfulTest("wy2", c, 5)


## yty1 Test
    def test_yty1_real_size1(self):
        
        c = self.__setupC(1,1, True)
        self.__methodSuccessfulTest("yty1", c, 1)

    def test_yty1_real_N20m4(self):
        c = self.__setupC(20,4, True)
        self.__methodSuccessfulTest("yty1", c, 2)

    def test_yty1_real_N20m20(self):
        c = self.__setupC(20,20, True)
        self.__methodSuccessfulTest("yty1", c, 5)

    def test_yty1_complex_size1(self):
        
        c = self.__setupC(1,1, False)
        self.__methodSuccessfulTest("yty1", c, 1)

    def test_yty1_complex_N20m4p2(self):
        c = self.__setupC(20,4, False)
        self.__methodSuccessfulTest("yty1", c, 2)

    def test_yty1_complex_N20m20p5(self):
        c = self.__setupC(20,20, False)
        self.__methodSuccessfulTest("yty1", c, 5)



## yty2 Test
    def test_yty2_real_size1(self):
        
        c = self.__setupC(1,1, True)
        self.__methodSuccessfulTest("yty2", c, 1)

    def test_yty2_real_N20m4(self):
        c = self.__setupC(20,4, True)
        self.__methodSuccessfulTest("yty2", c, 2)

    def test_yty2_real_N20m20(self):
        c = self.__setupC(20,20, True)
        self.__methodSuccessfulTest("yty2", c, 5)

    def test_yty2_complex_size1(self):
        
        c = self.__setupC(1,1, False)
        self.__methodSuccessfulTest("yty2", c, 1)

    def test_yty2_complex_N20m4p2(self):
        c = self.__setupC(20,4, False)
        self.__methodSuccessfulTest("yty2", c, 2)

    def test_yty2_complex_N20m20p5(self):
        c = self.__setupC(20,20, False)
        self.__methodSuccessfulTest("yty2", c, 5)



    def __setupC(self, N, m, real = False):
        func = generateHermetianT
        if real:
            func = generateT
        return Cholesky(func(N,m))
    def __methodSuccessfulTest(self, method, c, p):
        L = c.fact(method, p)
        T = L.dot(np.conj(L.T))[:, :c.m]
        self.assertTrue(testFactorization(T, L))

   ##TODO: Failed tests - when m > N, when there is not a valid method, when p < 1
     

    def tearDown(self):
        pass




if __name__ == '__main__':
    unittest.main()
