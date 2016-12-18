import sys
import numpy as np
import scipy as sp
from scipy import linalg
#from numpy import linalg as LA
#from toeplitz_decomp import *
#import matplotlib.pyplot as plt
from reconstruct1 import *
import os	
import mmap


mm=mmap.mmap(-1,256,mmap.MAP_PRIVATE)
np.set_printoptions(precision=2, suppress=True, linewidth=5000)
if len(sys.argv) < 8:
    print "Usage: %s filename num_rows num_columns offsetn offsetm sizen sizem" % (sys.argv[0])
    sys.exit(1)

num_rows=int(sys.argv[2])
num_columns=int(sys.argv[3])
offsetn=int(sys.argv[4])
offsetm=int(sys.argv[5])
sizen=int(sys.argv[6])
sizem=int(sys.argv[7])
nump=sizen

if offsetn>num_rows or offsetm>num_columns or offsetn+sizen>num_rows or offsetm+sizem>num_columns:
	print "Error sizes or offsets don't match"
	sys.exit(1)

a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')

#plt.subplot(1, 1, 1)
#plt.imshow(a.T, interpolation='nearest')
#plt.colorbar()
#plt.show()

pad=1
pad2=1
debug=0

neff=sizen+sizen*pad
meff=sizem+sizem*pad

a_input=np.zeros(shape=(neff,meff), dtype=complex)
a_input[:sizen,:sizem]=np.copy(a[offsetn:offsetn+sizen,offsetm:offsetm+sizem])
del a

a_input=np.where(a_input > 0, a_input, 0)
const=pad2*meff/2
a_input=np.sqrt(a_input)
if debug:
	print a_input,"after sqrt"

a_input[:sizen,:sizem]=np.fft.fft2(a_input,s=(sizen,sizem))


if debug:
	print a_input,"after first fft"


a_input[neff-(sizen/2-1):neff,0:sizem/2]=a_input[sizen/2+1:sizen,0:sizem/2]
a_input[0:sizen/2,meff-(sizem/2-1):meff]=a_input[0:sizen/2,sizem/2+1:sizem]
a_input[neff-(sizen/2-1):neff,meff-(sizem/2-1):meff]=a_input[sizen/2+1:sizen,1+sizem/2:sizem]
a_input[sizen/2:sizen,:sizem]=np.zeros(shape=(sizen/2,sizem))
a_input[:sizen/2,sizem/2:sizem]=np.zeros(shape=(sizen/2,sizem/2))

if debug:
	print a_input,"after shift"


#plt.subplot(1, 1, 1)
#plt.imshow(a.T, interpolation='nearest')
#plt.colorbar()
#plt.show()
#corr=np.zeros(shape=(neff, meff), dtype=complex)
#for i in xrange(0,neff):
#	for j in xrange(0,meff):		
#		temp = np.roll(a_input,j,axis=1)
#		corr[i,j]=signal.correlate(a_input, np.roll(temp,i,axis=0),mode='valid')[0,0] /(neff*meff+0j)	
#print corr,"corr"

a_input=np.fft.ifft2(a_input,s=(neff,meff))
if debug:
	print a_input,"after second fft"
a_input=np.power(np.abs(a_input),2)
if debug:
	print a_input,"after abs^2"
a_input=np.fft.fft2(a_input,s=(neff,meff)) 
if debug:
	print a_input,"after third fft"
	
comp=np.concatenate((a_input[:,:meff/2],np.zeros(shape=(neff,meff)),a_input[:,meff/2:meff]),axis=1)
#print comp

meff_f=meff+pad2*meff
epsilon=np.identity(meff_f)*10e-8
input_f=np.zeros(shape=(meff_f, sizen*meff_f), dtype=complex)
#for i in xrange(0,neff/2):
#	for j in xrange(i,neff/2):
#		if j>i:
#			rows=np.append(a_input[j-i,:meff-const],np.zeros(pad2*meff+const))
#			cols=np.append(np.append(a_input[j-i,0],a_input[j-i,const+1:][::-1]),np.zeros(pad2*meff+const))
#			input_f[i*meff_f:(i+1)*meff_f,j*meff_f:(j+1)*meff_f]=toeplitz(cols,rows)
#		else:
#			input_f[i*meff_f:(i+1)*meff_f,j*meff_f:(j+1)*meff_f]=toeplitz(np.conj(np.append(a_input[j-i,:meff-const],np.zeros(pad2*meff+const))))+epsilon

for j in xrange(0,neff/2):
	if j:
		rows=np.append(a_input[j,:meff-const],np.zeros(pad2*meff+const))
		cols=np.append(np.append(a_input[j,0],a_input[j,const+1:][::-1]),np.zeros(pad2*meff+const))
		input_f[0:meff_f,j*meff_f:(j+1)*meff_f]=sp.linalg.toeplitz(cols,rows)
	else:
		input_f[0:meff_f,j*meff_f:(j+1)*meff_f]=sp.linalg.toeplitz(np.conj(np.append(a_input[j,:meff-const],np.zeros(pad2*meff+const))))+epsilon


#input_f=np.conj(np.triu(input_f).T)+np.triu(input_f,1)
#print input_f[:,sizen*meff_f-1].T,"last column"
#print input_f[:,sizen*meff_f-1].shape
#for i in xrange(0,neff/2):
#	print input_f[(i+1)*meff_f-1,(neff/2-1)*meff_f:(neff/2)*meff_f].shape
#	print input_f[(i+1)*meff_f-1,(neff/2-1)*meff_f:(neff/2)*meff_f]
#np.save('100by100.npy',a_input/np.sqrt(neff*meff))
#if debug:
#	print input_f[0:meff+pad2*meff,0:(meff)], "blocks00"
#	print input_f[0:meff+pad2*meff,meff+pad2*meff:meff_f+(meff)],"blocks01"
#	print input_f[meff+pad2*meff:meff_f+(meff),0:meff+pad2*meff],"blocks10"
#	print input_f[0:meff+pad2*meff,2*meff_f:2*meff_f+meff_f],"blocks02"
#	print input_f[0:meff+pad2*meff,3*meff_f:3*meff_f+(meff)],"blocks03"
#print input_f[0:meff+pad2*meff,4*meff_f:4*meff_f+(meff)],"blocks04"
#print input_f[0:meff+pad2*meff,5*meff_f:5*meff_f+(meff)],"blocks05"
#print input_f[0:8,16:32], neff,meff_f
#print np.sum(input_f, axis=1),"sums"

#input_f=epsilon+input_f
#w, v = LA.eig(input_f)
#print w,"values"
#print v,"vectors"
#print np.sort(np.real(np.linalg.eigvals(input_f)))
#input_f=input_f+epsilon
#L=np.linalg.cholesky(input_f)
#L=np.conj(L.T)

path="processedData/gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s/" %(str(sizen),str(meff_f),str(offsetn),str(offsetm))
mkdir="mkdir "+path
os.system(mkdir)

filen=path+"gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s_toep.npy" %(str(sizen),str(meff_f),str(offsetn),str(offsetm))
np.save(filen,comp)

for rank in xrange(0,nump):
	size_node_temp=(sizen//nump)*meff_f
	size_node=size_node_temp
	if rank==nump-1:
		size_node = (sizen//nump)*meff_f + (sizen%nump)*meff_f
	start = rank*size_node_temp
	file_name=path+str(rank)+".npy"
	np.save(file_name,np.conj(input_f[:,start:start+size_node].T))
output_file="processedData/gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s.dat" %(str(sizen),str(meff_f),str(offsetn),str(offsetm))
output = np.memmap(output_file, dtype='complex', mode='w+', shape=(meff_f, sizen*meff_f),order='F')
output[:,:]=input_f[:meff_f,:]
del output
#print input_f[0:sizen*meff_f, sizem*meff_f:neff*meff_f]
mm.close()
if debug:
	pad=1
	u=toeplitz_blockschur(input_f[:neff/2*meff_f,:neff/2*meff_f],meff_f,pad)
	#print u[]
	#l=np.conj(u.T)
	#t=np.dot(l,u)
	print u[:,(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2]
	#lr=t[(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2,:].T
	#results=reconstruct_map(uc,lr,meff_f,pad)
	#print results,"reconstrcuted Toeplitz"
	#print np.append(l[:,l.shape[1]-1-meff_f:l.shape[1]-meff_f],l[:,l.shape[1]-1:l.shape[1]],axis=1),l.shape
