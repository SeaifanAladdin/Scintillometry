import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + "/../../")
import numpy as np

import timeit

if len(sys.argv) !=7:
    print "Six arguments needed: Thread, number of processors, number of blocks, size of block, blocking size, number of times to test"
    sys.exit(1)

def getArg(i):
    comma = ","
    dash = "-"
    arg = sys.argv[i]
    if dash in arg and comma in arg:
        print "Cannot have , and - in argument"
        sys.exit(1)
    if dash in arg:
        v = arg.split("-")
        return np.arange(int(v[0]), int(v[-1]) + 1)
    elif comma in arg:
        v = arg.split(",")
        return np.array(v)
    else:
        return np.array([arg])
        

thread = getArg(1)
npr = getArg(2)
n = getArg(3)
m = getArg(4)
p = getArg(5)
num = sys.argv[6] ## number of time to test


writeTo = currentdir
try:
    writeTo = os.environ["WRITE_TO"]
except:
    sys.stderr.write("Save location at: " + currentdir + "\n")


METHODS = ["seq", "wy1", "wy2", "yty1", "yty2", "numpy", "Niliou's seq"]


fi1 = "timingTest.py"
fi2 = "timingMPITest.py"

for threadi in thread:
    for npri in npr:
        for ni in n:
            for mi in m:
                for pi in p:
                    os.system("WRITE_TO='{4}' OMP_NUM_THREADS=8  mpirun -np 1 python {7} {0} {1} {2} {3} {5} {6}".format(ni, mi, pi, num, writeTo, threadi, npri, fi1))
                    os.system("WRITE_TO='{4}' OMP_NUM_THREADS=8  mpirun -np {6} python {7} {0} {1} {2} {3} {5} {6}".format(ni, mi, pi, num, writeTo, threadi, npri, fi2))

