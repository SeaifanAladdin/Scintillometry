# Scintillometry
Research on pulsar VLBI scintillometry. The purpose of this project is to decompose a toeplitz matrix using Schur's algoritm. Because this project will run on SciNet, a supercomputer, the program will need to have multi-threading and multi-core capabilities. 
Currently, the software works for a hermetian matrix with all five methods as decribed by Bereux.

To use, run 
$ python src/interface.py

The program will ask for the size of the matrix. It will then ask the size of the blocks. The size of the matrix must be an integer multiple of the size of the blocks.
Moving on, you must specify a method; seq, wy1, wy2, yty1, and yty2.
If you choose any except the first method, you will be asked for the blocking factor p, which must be greator than 1.

