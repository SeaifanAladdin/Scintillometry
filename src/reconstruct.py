import sys
import numpy as np


def reconstruct_map(uc,lr,meff_f,pad):
	if pad:
		n=uc.shape[0]/(meff_f)
	else:
		print "one round of padding necessary" 
		return 0
	m=meff_f/4
	print meff_f
	result=np.zeros(shape=(n*2,meff_f),dtype=complex)
	for i in xrange(0,2*n):
		if i<=(n-1):
			#print np.fliplr(np.roll(uc[i*(meff_f):i*(meff_f)+meff_f].T,2*m)).shape
			result[n-1-i,0:meff_f]=np.fliplr(np.roll(uc[i*(meff_f):i*(meff_f)+meff_f].T,2*m))
		elif i>n:
			result[i,0:meff_f]=np.roll(lr[(i-n-1)*(meff_f):(i-n-1)*(meff_f)+meff_f].T,2*m+1)
	return result
	
