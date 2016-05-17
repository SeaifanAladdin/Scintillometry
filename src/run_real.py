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

if len(sys.argv) != 10:
	if rank==0:
		print "Please pass in the following arguments: method n m p pad"
else:
	num_rows = int(sys.argv[2])
	num_columns=int(sys.argv[3])
	offsetn=int(sys.argv[4])
	offsetm=int(sys.argv[5])
	n = int(sys.argv[6])
	m = int(sys.argv[7])
	method = sys.argv[1]
	p = int(sys.argv[8])
	pad = sys.argv[9] == "1" or sys.argv[9] == "True"
	
	filename = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n,m*4, offsetn, offsetm) 
	file = filename + ".dat".format(n, m*4)
	folder = "processedData"
	cmd = "python extract_realData.py {} {} {} {} {} {} {}".format(FILE, num_rows, num_columns, offsetn, offsetm, n,m)
	os.system(cmd)


	T = np.zeros((2*m, 2*m), complex)
	toeplitz=None
	if rank==0:
		toeplitz1 = np.memmap("{0}/{1}".format(folder,file), dtype='complex', mode='r', shape=(4*m,4*n*m), order='F')
		toeplitz = np.zeros((4*m*n*(1 + pad), 4*n*m*(1 + pad)), complex)
	
	
		#for i in range(0,2*n,2):
		#	T_temp = toeplitz1[:2*m, 2*i*m:2*(i + 1)*m].copy()
		#	toeplitz[:2*m, (i/2)*2*m:2*(i/2 + 1)*m] = T_temp
		toeplitz[:4*m, :4*n*m] = toeplitz1
		if pad:
			T_temp = np.zeros((4*m,4*m),complex)
			for i in range(0, n):
				toeplitz[:4*m, (i + n)*4*m: 4*(i + 1 + n)*m] = T_temp

		for i in range(0,n*(1 + pad)):
			T_temp = toeplitz[:4*m,4*i*m:4*(i+1)*m]
			for j in range(0,n*(1 + pad) - i):
				k = j + i
				toeplitz[4*j*m:4*(j+1)*m,4*k*m:4*(k+1)*m] = T_temp
				toeplitz[4*k*m:4*(k+1)*m, 4*j*m:4*(j+1)*m] = np.conj(T_temp.T)
		fig = plt.figure()
		plt.imshow(np.abs(toeplitz))
		plt.colorbar()
		if not pad:
			plt.title("Toeplitz matrix with np = {0}, n = {1}, m = {2}".format(size, n, m*4))
		else:
			plt.title("Zero padded Toeplitz matrix with np = {0}, n = {1}, m = {2}".format(size, n, m*4))
		ax = fig.gca()
		ax.set_xticks(np.arange(0,4*n*m*(1 + pad),4*m))
		ax.set_yticks(np.arange(0,4*n*m*(1 + pad),4*m))
		ax.grid(True, which='both')
		if pad:
			plt.plot([0, 2*4*n*m], [4*n*m, 4*m*n], '-k')
			plt.plot([4*n*m, 4*m*n],[0, 2*4*n*m], '-k')
		plt.show()
		
	toeplitz = comm.bcast(toeplitz, root=0)	
	c = ToeplitzFactorizor(n,4*m, pad)
	for i in range(0, n//size):
		T = toeplitz[:4*m, 4*(rank+ i*size)*m:4*(rank + 1 + i*size)*m]
		T = np.conj(T.T)
		c.addBlock(T, rank + i*size)
	
	L = c.fact(method, p)
	
        #print np.max(np.abs(L.dot(np.conj(L.T))[0*m*n:2*m*n,:2*m*n] - T))
	
	
	if rank == 0:
		npL = cholesky(toeplitz, True)
		x = np.zeros((4*n*m, 1), complex)
		for i in range(n):
			x[4*i*m:(i+1)*4*m, :] = -np.conj(L[4*n*m*pad:,4*n*m*pad:]).T[i*4*m:(i+1)*4*m, 4*m*n- 2*m - 1:4*m*n - 2*m]

		#x[:, 0] = x[::-1,0] 

		resultpath_uc='results/'+filename+'_uc.npy'
		np.save(resultpath_uc, x) ## Might be conjugate
		fig = plt.figure()
		plt.subplot(1,2,1)
		ax = fig.gca()
		ax.set_xticks(np.arange(0,4*n*m*(1 + pad),4*m))
		ax.set_yticks(np.arange(0,4*n*m*(1 + pad),4*m))
		ax.grid(True, which='both')
		if pad:
			plt.plot([0, 2*4*n*m], [4*n*m, 4*m*n], '-k')
			plt.plot([4*n*m, 4*m*n],[0, 2*4*n*m], '-k')
		
		if pad:
			plt.imshow(np.abs(toeplitz - L.dot(np.conj(L.T))))
		else:
			plt.imshow(np.abs(toeplitz - L.dot(np.conj(L.T))))
		plt.colorbar()
		plt.title("Errors on the Toeplitz Matrix")
		plt.subplot(1,2,2)
		ax = fig.gca()
		ax.set_xticks(np.arange(0,4*n*m*(1 + pad),4*m))
		ax.set_yticks(np.arange(0,4*n*m*(1 + pad),4*m))
		ax.grid(True, which='both')
		if pad:
			plt.plot([0, 2*4*n*m], [4*n*m, 4*m*n], '-k')
			plt.plot([4*n*m, 4*m*n],[0, 2*4*n*m], '-k')
		plt.title("Factorized Toeplitz Matrix")
		plt.imshow(np.abs(L))
		plt.colorbar()
		plt.show()

