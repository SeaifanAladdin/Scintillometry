import os,sys
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from Factorize_parrallel import ToeplitzFactorizor
from scipy.linalg import cholesky

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

FILE = "gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined"

if len(sys.argv) != 6:
	if rank==0:
		print "Please pass in the following arguments: method n m p pad"
else:
	n = int(sys.argv[2])
	m = int(sys.argv[3])
	method = sys.argv[1]
	p = int(sys.argv[4])
	pad = sys.argv[5] == "1" or sys.argv[5] == "True"

	file = "gate0_numblock_{}_meff_{}_offsetn_100_offsetm_100.dat".format(n, m*4)
	folder = "processedData"
	cmd = "python extract_realData.py {} 2048 330 100 100 {} {}".format(FILE, n,m)
	os.system(cmd)


	T = np.zeros((2*m, 2*m), complex)
	toeplitz=None
	if rank==0:
		toeplitz1 = np.memmap("{0}/{1}".format(folder,file), dtype='complex', mode='r', shape=(4*m,4*n*m), order='F')
		toeplitz = np.zeros((2*m*n, 2*n*m), complex)
	
	
		for i in range(0,2*n,2):
			T_temp = toeplitz1[:2*m, 2*i*m:2*(i + 1)*m].copy()
			toeplitz[:2*m, (i/2)*2*m:2*(i/2 + 1)*m] = T_temp

		for i in range(0,n):
			T_temp = toeplitz[:2*m,2*i*m:2*(i+1)*m]
			for j in range(0,n - i):
				k = j + i
				toeplitz[2*j*m:2*(j+1)*m,2*k*m:2*(k+1)*m] = T_temp
				toeplitz[2*k*m:2*(k+1)*m, 2*j*m:2*(j+1)*m] = np.conj(T_temp.T)
		
 		#plt.subplot(1,2,1)
		#plt.imshow(np.abs(toeplitz))
		#plt.show()
		
	toeplitz = comm.bcast(toeplitz, root=0)	
	c = ToeplitzFactorizor(n,2*m, pad)
	for i in range(0, n//size):
		T = toeplitz[:2*m, 2*(rank+ i*size)*m:2*(rank + 1 + i*size)*m]
		T = np.conj(T.T)
		c.addBlock(T, rank + i*size)
	
	L = c.fact(method, p)
	
        #print np.max(np.abs(L.dot(np.conj(L.T))[0*m*n:2*m*n,:2*m*n] - T))
	
	
	if rank == 0:
		npL = cholesky(toeplitz, True)
		plt.subplot(1,2,1)
		if pad:
			plt.imshow(np.abs(toeplitz - L.dot(np.conj(L.T))[2*m*n:2*2*m*n, 2*m*n:2*2*m*n]))
		else:
			plt.imshow(np.abs(toeplitz - L.dot(np.conj(L.T))))
		print L.shape
		plt.colorbar()
		plt.title("Errors on the Toeplitz Matrix")
		plt.subplot(1,2,2)
		
		plt.title("Factorized Toeplitz Matrix")
		plt.imshow(np.abs(L))
		plt.colorbar()
		plt.show()

