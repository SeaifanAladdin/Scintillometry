import os,sys
from mpi4py import MPI
import numpy as np
from Factorize_parrallel import ToeplitzFactorizor

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

FILE = "gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined"

if len(sys.argv) != 8:
	if rank==0:
		print "Please pass in the following arguments: method offsetn offsetm n m p pad"
else:
	offsetn=int(sys.argv[2])
	offsetm=int(sys.argv[3])
	n = int(sys.argv[4])
	m = int(sys.argv[5])
	method = sys.argv[1]
	p = int(sys.argv[6])
	pad = sys.argv[7] == "1" or sys.argv[7] == "True"
	
	if not os.path.exists("processedData/"):	
		os.makedirs("processedData/")
	folder = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n, m*4, offsetn, offsetm)
	filename = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}_toep".format(n,m*4, offsetn, offsetm) 
	file = filename + ".npy".format(n, m*4)
	T = np.zeros((2*m, 2*m), complex)
	toeplitz=None

	c = ToeplitzFactorizor(folder, n,4*m, pad)
	for i in range(0, n*(1 + pad)//size):
		c.addBlock(rank + i*size)
	c.fact(method, p)
	
	'''
	L = np.zeros((4*m*n*(1 + pad),4*m*n*(1 + pad)), complex)
	
	import matplotlib.pyplot as plt
	if rank == 0:
		for i in range(n*(1 + pad)):
			for j in range(n*(1 + pad)): 
				path = "results/{0}/L_{1}-{2}.npy".format(folder, i,j)     
				
				if os.path.isfile(path):
					Ltemp = np.load(path)
					L[4*m*j: 4*m*(j + 1), 4*m*i:4*m*(i + 1)] = Ltemp
	    
		np.save('results/'+filename+'.npy',L)
		plt.imshow(np.abs(L)**.001)
		plt.show()
        
	    		
		x = np.zeros((4*n*m, 1), complex)
		for i in range(n):
			x[4*i*m:(i+1)*4*m, :] = -np.conj(L[4*n*m*pad:,4*n*m*pad:]).T[i*4*m:(i+1)*4*m, 4*m*n- 2*m - 1:4*m*n - 2*m]

		#x[:, 0] = x[::-1,0] 

		resultpath_uc='results/'+filename+'_uc.npy'
		np.save(resultpath_uc, x) ## Might be conjugate
		
    '''