import numpy as np
#import matplotlib.pyplot as plt
import sys
import timeit

import os
numThread = os.environ["OMP_NUM_THREADS"]
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

SETUP1 ="""n = {0}; m = {1}; p = {2};
T = createBlockedToeplitz(n, m)
c = ToeplitzFactorizor(T, m)
"""

SETUP2="""n, m = {0}, {1};
T = createBlockedToeplitz(n, m)"""

SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"

methods = [SEQ, WY1, WY2, YTY1, YTY2, "numpy", "Niliou's seq"]

n = 300
m = 4
p = 2


thread_arr = np.arange(1, numThread + 1)
timeMethods = np.zeros((7, 2,len(thread_arr)) )
for th in thread_arr:
    os.environ["OMP_NUM_THREADS"] = th
    print "thread = {0}".format(th)
    i = th - 1
    setup = SETUP + SETUP1.format(n,m,p)
    setup2 = SETUP + SETUP2.format(n, m)
    for j in range(len(methods)):
        if j < 5:
            t = timeit.repeat("c.fact(\"{0}\", p)".format(methods[j]), setup, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        if j == 5:
            t = timeit.repeat("np.linalg.cholesky(T)", setup2, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        elif j == 6:
            t = timeit.repeat("td.toeplitz_blockschur(T, {0}, 0)".format(n), setup2, number= num)
            timeMethods[j, :, i] = np.mean(t), np.std(t)

for j in range(len(methods)):
    np.savetxt("/scratch2/p/pen/seaifan/results/{0}-VaryingThreads-n{1}m{2}p{3}.txt".format(methods[j], n, m, p), timeMethods[j])
