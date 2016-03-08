# Scintillometry

##Synopsis
The purpose of this project is to decompose a toeplitz matrix using Schur's algoritm. Because this project will run on SciNet, a supercomputer, the program will need to have multi-threading and multi-core capabilities. 

##Example
```
$ python src/interface.py 
Choose the size of your block:
2
Choose the number of blocks:
2
Your Toeplit Matrix is: 
[[ 34.69+0.j   5.00-2.j  19.25+0.j   1.00-9.j]
 [  5.00+2.j  34.69+0.j   1.00+9.j  19.25+0.j]
 [ 19.25-0.j   1.00-9.j  34.69+0.j   5.00-2.j]
 [  1.00+9.j  19.25-0.j   5.00+2.j  34.69+0.j]]

Choose a method among: seq wy1 wy2 yty1 yty2
wy2
Choose your p-factor (Not neccesary for seq):
1
The cholesky decomposition of the generated Toeplitz Matrix:
[[ 5.89+0.j    0.00+0.j    0.00+0.j    0.00+0.j  ]
 [ 0.85+0.34j  5.82+0.j    0.00+0.j    0.00+0.j  ]
 [ 3.27+0.j   -0.31-1.36j -4.70-0.j    0.00+0.j  ]
 [ 0.17+1.53j  3.20-0.21j -1.09+1.57j -4.29-0.j  ]]
```
##Motivation
When radio waves from puslars passes through the ISM, Interstellar Medium, a scintillating speckle pattern is created. To understand the properties of the ISM, and to do further research on pulsars, we require the phase and magnitude of the observed scintillation pattern. The purpose of this project is to obtain both the phase and magnitude by being able to factorize a toeplitz matrix using Schur's algorithm. 

##Tests
In the src/Tests folder are the unittest code which tests the software. The python file ToeplitzFactorizorTest.py contains 30 tests, which includes testing different methods, sizes, and block sizes.
```
$python src/Test/ToeplitzFactorizorTest.py 
..............................
----------------------------------------------------------------------
Ran 30 tests in 0.039s

OK

```

If you would like to make your own tests, use the function testFactorization from func.py. This function takes in your toeplitz matrix **T** and the computed factorized matrix **L** and confirms whether **T** = **L** **L^t** where t is the conjugate transpose.
```
$ cd src/
$ python
Python 2.7.10 (default, Sep  8 2015, 17:20:17) 
[GCC 5.1.1 20150618 (Red Hat 5.1.1-4)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> from func import *
>>> from ToeplitzFactorizor import *
>>> T = createBlockedToeplitz(20, 4)
>>> c = ToeplitzFactorizor(T, 20)
>>> L = c.fact("yty2", 2)
>>> print "Factorization works: " + str(testFactorization(T, L))
Factorization works: True
```

