import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from numpy import linalg as LA
from mpi4py import MPI
#import h5py 
import time
from sp import multirate
from parallel_decom_pading import *
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.pylab as plt
import matplotlib.cm as cm
from reconstruct import *
import re

if len(sys.argv) < 2:
    print "Usage: %s filename(withoutextention)" % (sys.argv[0])
    sys.exit(1)
filename=sys.argv[1]
filename_toep='processedData/'+filename+"/"+filename+'_toep.npy'
resultpath_uc='results/'+filename+'_toep_uc.npy'

matchObj = re.search('meff_(\d*)',filename) 
if matchObj:    
	meff_f=matchObj.group(1)
else:
	sys.exit(1)
uc=np.load(resultpath_uc)
#uc=uc.T
print uc.shape,int(meff_f)
lr=np.zeros_like(uc)
results=reconstruct_map(uc,lr,int(meff_f),1)

cj=np.load(filename_toep)
print results.shape,cj.shape
A = np.log10(np.power(np.abs(results),2))
#cg = np.log10(np.power(np.abs(cj[:cj.shape[0]/2,:]),2))
#vmin = A.mean()-2.*A.std()
#vmax = A.mean()+A.std()
#print cg.shape,A.shape
plt.figure(1)
plt.subplot(211)
plt.imshow(A, aspect='auto', cmap=cm.Greys, interpolation='nearest', vmin=-4, origin='lower')
plt.colorbar()
plt.ylabel("tau")
plt.title("log |cholesky|^2")
plt.xlabel("fd")
plt.subplot(212)
plt.imshow(np.log10(np.power(np.abs(cj),1)), aspect='auto', cmap=cm.Greys, interpolation='nearest', vmin=-4, origin='lower')
plt.colorbar()
plt.title("|Input matrix| (the one Toeplitz matrix is constructed from)")
plt.ylabel("tau")
plt.xlabel("fd")
plt.subplot(211)
plt.show()
