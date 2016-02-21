import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit


SETUP = """import os,sys,inspect;
import toeplitz_decomp as td;
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())));
parentdir = os.path.dirname(currentdir); 
sys.path.insert(0,parentdir); 
sys.path.insert(0, parentdir + \"/Exceptions\");
from ToeplitzFactorizor import ToeplitzFactorizor;
import numpy as np;
from func import createT, generateHermetianM;

"""

SETUP1 ="""N = {0}; m = {1}; p = {2};
T = createT(generateHermetianM(N), m);
c = ToeplitzFactorizor(T)
"""

SETUP2="""N={0};
T = generateHermetianM(N)"""

SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"

methods = [SEQ, WY1, WY2, YTY1, YTY2, "numpy", "Niliou's seq"]

n = 10

timeMethods = np.zeros((7, 2, n - 1))


num = 10
N_arr = np.arange(1,n)*2
for N in N_arr:
    print "N = {}".format(N)
    i = N/2 - 1
    setup = SETUP + SETUP1.format(N,2,2)
    setup2 = SETUP + SETUP2.format(N)
    for j in range(len(methods)):
        if j < 5:
            t = timeit.repeat("c.fact(\"{}\", p)".format(methods[j]), setup, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        if j == 5:
            t = timeit.repeat("np.linalg.cholesky(T)", setup2, number = num)
            timeMethods[j, : ,i] = np.mean(t), np.std(t)
        elif j == 6:
            t = timeit.repeat("td.toeplitz_blockschur(T, 2, 0)", setup2, number= num)
            timeMethods[j, :, i] = np.mean(t), np.std(t)

for j in range(len(methods)):
    np.savetxt("{0}VaryingN.txt".format(methods[j]), timeMethods[j])
    
plt.figure()
plt.title("Execution time per method with n = 2, p =2")
plt.ylabel("Execution Time (s)")
plt.xlabel("Size of matrix")
for j in range(len(methods)):
    plt.errorbar(N_arr, timeMethods[j][0], fmt="-o",yerr=timeMethods[j][1], label=methods[j])
plt.errorbar(N_arr, timeMethods[5][0], fmt="-o", yerr=timeMethods[5][1], label="Numpy")
plt.errorbar(N_arr, timeMethods[6][0], fmt="-o", yerr=timeMethods[6][1], label="Nilou's code")

plt.legend(loc=2)
plt.savefig("toeplitzFactorizeVaryN.png")
plt.show()
