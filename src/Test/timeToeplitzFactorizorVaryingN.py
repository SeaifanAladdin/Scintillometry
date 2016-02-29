import numpy as np
##import matplotlib.pyplot as plt
import sys
import timeit

import threading
numThread = threading.active_count()
print numThread


SETUP = """import os,sys,inspect;
import toeplitz_decomp as td;
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())));
parentdir = os.path.dirname(currentdir); 
sys.path.insert(0,parentdir); 
sys.path.insert(0, parentdir + \"/Exceptions\");
from ToeplitzFactorizor import ToeplitzFactorizor;
import numpy as np;
from func import createToeplitz;

"""

SETUP1 ="""N = {0}; m = {1}; p = {2};
T = createToeplitz(N);
c = ToeplitzFactorizor(T, m)
"""

SETUP2="""N={0};
T = createToeplitz(N)"""

SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"

methods = [SEQ, WY1, WY2, YTY1, YTY2, "numpy", "Niliou's seq"]

p = 3
m = 10
num = 10
N_i = 10
N_f = 30
N_step = m

timeMethods = np.zeros((7, 2, (N_f - N_i)/N_step + 1) )

N_arr = np.arange(N_i,N_f + N_step, N_step)
for N in N_arr:
    print "N = {}".format(N)
    i = float(N - N_i)/N_step
    i = int(i)
    setup = SETUP + SETUP1.format(N,m,p)
    setup2 = SETUP + SETUP2.format(N)
    for j in range(len(methods)):
        if j < 5:
            t = timeit.repeat("c.fact(\"{}\", p)".format(methods[j]), setup, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        if j == 5:
            t = timeit.repeat("np.linalg.cholesky(T)", setup2, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        elif j == 6:
            t = timeit.repeat("td.toeplitz_blockschur(T, {}, 0)".format(m), setup2, number= num)
            timeMethods[j, :, i] = np.mean(t), np.std(t)

for j in range(len(methods)):
    np.savetxt("results/thread{}-{}VaryingNn{}p{}.txt".format(numThread, methods[j], m, p), timeMethods[j])
    
##plt.figure()
##plt.title("Execution time per method with n = {}, p = {}".format(m, p))
##plt.ylabel("Execution Time (s)")
##plt.xlabel("Size of matrix")
##for j in range(len(methods)):
##    plt.errorbar(N_arr, timeMethods[j][0], fmt="-o",yerr=timeMethods[j][1], label=methods[j])
##    
##plt.legend(loc=2)
##plt.savefig("toeplitzFactorizen{}p{}VaryN{}-{}-{}.png".format(m,p, N_i, N_f, N_step))
##plt.show()
