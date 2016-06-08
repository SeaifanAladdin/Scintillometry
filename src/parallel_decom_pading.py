import numpy as np
from mpi4py import MPI
def block_toeplitz_par(file_name,n,b,pad):
	comm = MPI.COMM_WORLD
	size  = comm.Get_size()
	rank = comm.Get_rank()
	data_type_size=16 #complex has 16 bytes
	size_node_temp=(n//size)*b
	size_node=size_node_temp
	if rank==size-1:
		size_node = (n//size)*b + (n%size)*b
	start = rank*size_node_temp
	file_offset = int(start * data_type_size*b)
	end = min(start+size_node, n*b)	
	g1=np.zeros(shape=(b,size_node), dtype=complex)
	g2=np.zeros(shape=(b,size_node), dtype=complex)
	temp=np.zeros(shape=(b,b), dtype=complex)
	uc=np.zeros(shape=(0,1), dtype=complex)
	name=file_name+str(rank)+".npy"
	a=np.load(name)
	print a
	#a = np.memmap(file_name, dtype='complex', mode='r',offset=file_offset,shape=(b,size_node),order='F')
	#a=np.load(file_name)[0:b,start:start+size_node]
	#if the first matrix is toeplitz:
	#firstblock=inv(toeplitz_decomp(np.array(a[0:b,0]).reshape(-1,).tolist()))
	if rank==0:
		firstblock=np.linalg.inv(np.linalg.cholesky(a[:b,:b]))
	else:
		firstblock=np.zeros(shape=(b,b), dtype=complex)
	firstblock=comm.bcast(firstblock ,root=0)		
	for j in xrange(0,size_node/b): 
		g2[:,j*b:(j+1)*b]= -np.dot(firstblock,a[0:b,j*b:(j+1)*b]) 
		g1[:,j*b:(j+1)*b]= -g2[:,j*b:(j+1)*b]
	if rank==size-1 and pad==0:
		uc=g1[:,end-start-b/2-1:end-start-end-start-b/2]
	empty=0
	del a
	#print firstblock,rank
	for i in xrange(1,n+n*pad):    
		global_end_g1=min(n*b,(n+n*pad-i)*b)
		start_g1=start
		if (global_end_g1<end and start<global_end_g1):
			end_g1=global_end_g1	
		elif (global_end_g1<end and start>=global_end_g1):
			empty=1
			g1=np.zeros_like(g1)
			end_g1=start
		else: 
			end_g1=end
		length_g=end_g1-start_g1
		if  rank !=size-1:
			comm.Recv(temp,source=rank+1,tag=i*size+rank)
		if  rank !=0: 
			data=np.copy(g2[:,0:b])
			comm.Send(data,dest=rank-1,tag=i*size+rank-1)
		g2[:,0:size_node-b]=g2[:,b:size_node]
		if  rank !=size-1:
			g2[:,size_node-b:size_node]=temp
		if (i<n*pad+1) and (rank==size-1):
			g2[:,size_node-b:size_node]=np.zeros(shape=(b,b))
		for j in xrange(0,b):
			if rank==0:
				g0_1=np.copy(g1[j,j])
				g0_2=np.copy(g2[:,j])
			else:
				g0_1=np.zeros(shape=(1,1), dtype=complex)
				g0_2=np.zeros(shape=(b,1), dtype=complex)
			g0_1=comm.bcast(g0_1 ,root=0)		
			g0_2=comm.bcast(g0_2,root=0)
			if empty:
				continue
			sigma=np.dot(np.conj(g0_2.T),g0_2)
			alpha=-np.sign(g0_1)*np.sqrt(g0_1**2.0 - sigma)
			z=g0_1+alpha
			if np.count_nonzero(g2[:])==0 or z==0:
				g2[:,0:length_g]=-g2[:,0:length_g]
				continue		
			x2=-np.copy(g0_2)/np.conj(z)
			beta=(2*z*np.conj(z))/(np.conj(z)*z-sigma)
			if rank==0 :
				g1[j,j]=-alpha
				g2[:,j]=0
				v=np.copy(g1[j,j+1:length_g]+np.dot(np.conj(x2.T),g2[:,j+1:length_g]))
				g1[j,j+1:length_g]=g1[j,j+1:length_g]-beta*v
				v=np.reshape(v,(1,v.shape[0]))
				x2=np.reshape(x2,(1,x2.shape[0]))
				g2[:,j+1:length_g]=-g2[:,j+1:length_g]-beta*np.dot(x2.T,v)
			else:
				v=np.copy(g1[j,:]+np.dot(np.conj(x2.T),g2))
				g1[j,:]=g1[j,:]-beta*v
				v=np.reshape(v,(1,v.shape[0]))
				x2=np.reshape(x2,(1,x2.shape[0]))
				g2=-g2-beta*np.dot(x2.T,v)
		if i>=n*pad and end_g1==global_end_g1 and length_g>0:
			#print uc.shape, length_g, rank
			uc=np.concatenate((uc,g1[:,length_g-b/2-1:length_g-b/2]),axis=0)
			#print l.shape
	uc = comm.gather(uc, root=0)
	if rank==0:
		uc=np.concatenate(uc[::-1], axis=0)
	return uc
