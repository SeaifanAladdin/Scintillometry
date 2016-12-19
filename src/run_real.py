import os,sys
from mpi4py import MPI
import numpy as np
from Factorize_parrallel import ToeplitzFactorizor

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

FILE = "gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined"

if len(sys.argv) != 8 and len(sys.argv) != 9:
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
	
	detailedSave = False
	if len(sys.argv) == 9:
	    detailedSave = sys.argv[8] == "1" or sys.argv[8] == "True"
	
	if not os.path.exists("processedData/"):	
		os.makedirs("processedData/")
	folder = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n, m*4, offsetn, offsetm)
	
	c = ToeplitzFactorizor(folder, n,4*m, pad, detailedSave)
	for i in range(0, n*(1 + pad)//size):
		c.addBlock(rank + i*size)
	c.fact(method, p)



