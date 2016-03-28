import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + "/../../")
import numpy as np

import timeit

if len(sys.argv) !=4:
    print "Three arguments needed: Thread, size of matrix, number of times to test"
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
N = getArg(2)
num = sys.argv[3] ## number of time to test


writeTo = currentdir
try:
    writeTo = os.environ["WRITE_TO"]
except:
    sys.stderr.write("Save location at: " + currentdir + "\n")


fi1 = "timingTest.py"
for Ni in N:
     for threadi in thread:
          os.system("WRITE_TO='{3}' OMP_NUM_THREADS={0}  mpirun -np 1 python {4} {0} {1} {2}".format(threadi, Ni, num, writeTo, fi1))

