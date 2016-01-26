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
Your generated Toeplitz Matrix:
⎡       345.0          249.0 + 86.0⋅ⅈ⎤ 
⎢                                    ⎥
⎢249.0 - -86.0⋅(-1)⋅ⅈ      280.0     ⎥
⎢                                    ⎥
⎢289.0 - -56.0⋅(-1)⋅ⅈ  309.0 + 10.0⋅ⅈ⎥
⎢                                    ⎥
⎣   234.0 + 12.0⋅ⅈ     230.0 + 95.0⋅ⅈ⎦

Choose a method among: seq wy1 wy2 yty1 yty2
wy1
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
