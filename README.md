# Scintillometry

##Synopsis
The purpose of this project is to decompose a toeplitz matrix using Schur's algoritm. Because this project will run on SciNet, a supercomputer, the program will need to have multi-threading and multi-core capabilities. 

##Example
```
$ python src/interface.py
Choose the size of your matrix:
4
Choose the size of your block:
2
Your randomly generated toeplitz matrix consists of 2 2x2 blocks
Those blocks are...
Block 1
⎡       345.0          249.0 + 86.0⋅ⅈ⎤
⎢                                    ⎥
⎣249.0 - -86.0⋅(-1)⋅ⅈ      280.0     ⎦
Block 2
⎡289.0 - -56.0⋅(-1)⋅ⅈ  309.0 + 10.0⋅ⅈ⎤
⎢                                    ⎥
⎣   234.0 + 12.0⋅ⅈ     230.0 + 95.0⋅ⅈ⎦

Therefore, your toeplitz matrix is:
⎡       345.0          249.0 + 86.0⋅ⅈ⎤
⎢                                    ⎥
⎢249.0 - -86.0⋅(-1)⋅ⅈ      280.0     ⎥
⎢                                    ⎥
⎢289.0 - -56.0⋅(-1)⋅ⅈ  309.0 + 10.0⋅ⅈ⎥
⎢                                    ⎥
⎣   234.0 + 12.0⋅ⅈ     230.0 + 95.0⋅ⅈ⎦

Choose a method among: seq wy1 wy2 yty1 yty2
wy2
Choose your p-factor (Not neccesary for seq):
2
The cholesky decomposition of the generated Toeplitz Matrix:
⎡       18.57                   0                0          0   ⎤
⎢                                                               ⎥
⎢13.41 - -4.63⋅(-1)⋅ⅈ         8.88               0          0   ⎥
⎢                                                               ⎥
⎢15.56 - -3.01⋅(-1)⋅ⅈ  9.74 - -2.44⋅(-1)⋅ⅈ     -26.4        0   ⎥
⎢                                                               ⎥
⎣   12.6 + 0.65⋅ⅈ         7.22 + 3.15⋅ⅈ     1.02 + 2.7⋅ⅈ  -15.04⎦
```
##Motivation
When radio waves from puslars passes through the ISM, Interstellar Medium, a scintillating speckle pattern is created. To understand the properties of the ISM, and to do further research on pulsars, we require the phase and magnitude of the observed scintillation pattern. The purpose of this project is to obtain both the phase and magnitude by being able to factorize a toeplitz matrix using Schur's algorithm. 

##Tests
In the src/Tests folder are the unittest code which tests the software. The python file ToeplitzFactorizorTest.py contains 30 tests, which includes testing different methods, sizes, and block sizes.
```
$ python src/Test/ToeplitzFactorizorTest.py 
..............................
----------------------------------------------------------------------
Ran 30 tests in 1.469s

OK
```

If you would like to make your own tests, use the function testFactorization from func.py. This function takes in your toeplitz matrix **T** and the computed factorized matrix **L** and confirms whether **T** = **L** **L^t** where t is the conjugate transpose.
```
>>> from func import *
>>> from ToeplitzFactorizor import *
>>> T = generateHermetianT(20, 4)
>>> c = ToeplitzFactorizor(T)
>>> L = c.fact("wy2", 2)
>>> print "Factorization works: " + str(testFactorization(T, L))
Factorization works: True

```

