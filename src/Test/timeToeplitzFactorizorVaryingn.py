import numpy as np
#import matplotlib.pyplot as plt
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



num = 10
n_i = 1
n_f = 10

def fact(n):
    if n <= 1:
        return 1
    else:
        return n*fact(n - 1)

def getMultiples(n):
    m = []
    for i in range(n, 0, -1):
        if n%i == 0:
            m.append(i)
    return np.array(m)

n_max = 3
N = fact(n_max)
print N
p = 2


n_arr = getMultiples(N)
n_arr.sort()
timeMethods = np.zeros((7, 2,len(n_arr)) )
for n in n_arr:
    print "n = {}".format(n)
    i = np.where(n_arr == n)[0]
    setup = SETUP + SETUP1.format(N,n,p)
    setup2 = SETUP + SETUP2.format(N)
    for j in range(len(methods)):
        if j < 5:
            t = timeit.repeat("c.fact(\"{}\", {})".format(methods[j], p), setup, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        if j == 5:
            t = timeit.repeat("np.linalg.cholesky(T)", setup2, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        elif j == 6:
            t = timeit.repeat("td.toeplitz_blockschur(T, {}, 0)".format(n), setup2, number= num)
            timeMethods[j, :, i] = np.mean(t), np.std(t)

for j in range(len(methods)):
    np.savetxt("results/thread{}-{}VaryingnN{}p{}.txt".format(numThread, methods[j], N, p), timeMethods[j])
    
##plt.figure()
##plt.title("Execution time per method with N = {}, p = {}".format(N, p))
##plt.ylabel("Execution Time (s)")
##plt.xlabel("Block size")
##for j in range(len(methods)):
##    plt.errorbar(n_arr, timeMethods[j][0], fmt="-o",yerr=timeMethods[j][1], label=methods[j])
##    
##plt.legend()
##plt.savefig("toeplitzFactorizeN{}p{}Varyn.png".format(N, p))
##plt.show()
