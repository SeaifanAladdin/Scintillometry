import numpy as np



import os,sys,inspect;
import toeplitz_decomp as td;
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())));
parentdir = os.path.dirname(currentdir); 
sys.path.insert(0,parentdir); 
sys.path.insert(0, parentdir + "/Exceptions");
import numpy as np;
from func import createBlockedToeplitz;



if len(sys.argv) != 4:
    print "Usage: %s filename(withoutextention)" % (sys.argv[0])
    sys.exit(1)

thread=int(os.environ["OMP_NUM_THREADS"])
n = int(sys.argv[1])
m = int(sys.argv[2])
p = int(sys.argv[3])

Methods = ["non-parallelized", "parallelized"]

methods = np.empty(2)
SETUP = """module purge;
module load intel/15.0.2 openmpi/intel/1.6.4 gcc python;
"""

from time import time
start=time()
os.system(SETUP + "OMP_NUM_THREADS={0} mpirun -np {1} python ../Factorize_parrallel.py seq {2} {3}".format(t, n, m, p))
end = time()
methids[0] = end - start

start = time()
os.system(SETUP + "OMP_NUM_THREADS={0} mpirun -np {1} python ../ToeplitzFactorizor.py seq {1} {2} {3}".format(t, n, m, p))
end = time()
methods[1] = end- start


np.savetxt("/scratch2/p/pen/seaifan/results/seqMPI_thread{0}_n{1}_m{2}_p{3}".format(thread, n, m, p), methods)
