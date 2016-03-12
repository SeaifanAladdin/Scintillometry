import numpy as np



import os,sys,inspect;
import toeplitz_decomp as td;
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())));
parentdir = os.path.dirname(currentdir); 
sys.path.insert(0,parentdir); 
sys.path.insert(0, parentdir + "/Exceptions");
import numpy as np;
from func import createBlockedToeplitz;



if len(sys.argv) != 5:
    print "Usage: %s filename(withoutextention)" % (sys.argv[0])
    sys.exit(1)

thread=int(sys.argv[1])
n = int(sys.argv[2])
m = int(sys.argv[3])
p = int(sys.argv[4])

threads = np.arange(1, thread + 1)

singleCore = np.zeros(threads.shape)
multiCore = np.zeros(threads.shape)

from time import time
for t in threads:
    start=time()
    os.system("module load mpi; OMP_NUM_THREADS={0} mpirun --np {1} python ../Factorize_parrallel.py seq {2} {3}".format(t, n, m, p))
    end = time()
    singleCore[t - 1] = end - start

    start = time()
    os.system("module load mpi; OMP_NUM_THREADS={0} mpirun --np {1} python ../ToeplitzFactorizor.py seq {1} {2} {3}".format(t, n, m, p))
    end = time()
    multiCore[t - 1] = end- start


np.savetxt("/scratch2/p/pen/seaifan/results/seqSingleMPI_thread{0}_n{1}_m{2}_p{3}".format(thread, n, m, p), singleCore)
np.savetxt("/scratch2/p/pen/seaifan/results/seqMultiMPI_thread{0}_n{1}_m{2}_p{3}".format(thread, n, m, p), multiCore)
