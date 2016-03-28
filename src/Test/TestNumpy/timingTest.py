import os, sys, inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + "/../../")

METHODS = ["seq", "wy1", "wy2", "yty1", "yty2", "numpy", "Niliou's seq"]

RESULTS = "/results"

def timeNumpy(N, num=3):
    from func import createBlockedToeplitz, testFactorization
    import timeit
    SETUP = """import numpy as np
N = {0};
A = np.random.rand(N,N)
Ai = np.random.rand(N,N)
M = np.array(A, complex) + Ai
A = np.random.rand(N,N)
Ai = np.random.rand(N,N)
M2 = np.array(A, complex) + Ai
"""
    setup = SETUP.format(N)
    t = timeit.repeat('np.dot(M,M2)', setup, number = num)
    return np.mean(t), np.std(t)



num = int(sys.argv[3]) ## number of time to test
thread = int(sys.argv[1])
N = int(sys.argv[2])

writeTo = currentdir
try:
    writeTo = os.environ["WRITE_TO"]
except:
    sys.stderr.write("Save location at: " + currentdir + "\n")

path = writeTo + RESULTS

data = np.array([[thread,N, None, None]])
if not os.path.exists(path):
    os.makedirs(path)
    

t, t_err = timeNumpy(N, num)

data[0][-2] = t
data[0][-1] = t_err
with open(path + "/NumpyDot.txt",'a') as f_handle:
    np.savetxt(f_handle,data)
