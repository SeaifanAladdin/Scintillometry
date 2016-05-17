import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib
from reconstruct import *
import re



matplotlib.rcParams.update({'font.size': 13})


if len(sys.argv) < 2:
    print "Usage: %s filename(withoutextention)" % (sys.argv[0])
    sys.exit(1)
filename=sys.argv[1]
filename_toep='processedData/'+filename+'_toep.npy'
filename_dynamic='processedData/'+filename+'_dynamic.npy'
resultpath_uc='results/'+filename+'_uc.npy'

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

#dx_ds, dy_ds = 19.7, 3906.25

# generate 2 2d grids for the x & y bounds
#y_ds, x_ds = np.mgrid[slice(0, 6501 + dy_ds, dy_ds),
#                slice(0, 8000000+ dx_ds, dx_ds)]

#cjj=np.fft.ifft2(cj)
#resultss=np.fft.ifft2(results)
#resultss=np.multiply(resultss,np.conj(resultss))
#corr=np.fft.fft2(resultss) *resultss.shape[0]*resultss.shape[1]
#print np.sum(cj[:510,:20]-corr[:510,:20]), cj[20,0], corr[20,0]
#corr=signal.fftconvolve(np.conj(results[::-1,::-1]),results)
#corr=np.append(corr[:,corr.shape[1]-corr.shape[1]/8:corr.shape[1]], corr[:,:corr.shape[1]/8],axis=1)[:corr.shape[0]/4,:]
bina=1
binb=1
#a_num=corr.shape[0]
#b_num=corr.shape[1]
#a_vv = np.copy(corr.reshape(a_num//bina, bina, b_num//binb, binb))
#corr=np.mean(np.mean(a_vv,axis=3),axis=1)
dynamic=np.load(filename_dynamic).T
dynamic=np.abs(dynamic)[:dynamic.shape[0]/2,:dynamic.shape[1]/2]



#a=dynamic.shape[0]
#b=dynamic.shape[1]
#bina=8
#binb=2
#a_view = dynamic.reshape(a//bina, bina, b//binb, binb)
#print a_view.shape,"heh"
#a_view=a_view.mean(axis=3).mean(axis=1)
#print a_view.shape,"shaoe"
#print results.shape,cj.shape
A = np.abs(results)
A=np.append(A[:,7*A.shape[1]/8:A.shape[1]], A[:,:A.shape[1]/8],axis=1)[:A.shape[0]/4,:]
bina=1
binb=1
a_num=A.shape[0]
b_num=A.shape[1]
a_vv = np.copy(A.reshape(a_num//bina, bina, b_num//binb, binb))
A=np.mean(np.mean(a_vv,axis=3),axis=1)
cj=np.append(cj[:,7*cj.shape[1]/8:cj.shape[1]], cj[:,:cj.shape[1]/8],axis=1)[:cj.shape[0]/4,:]

#cg = np.log10(np.power(np.abs(cj[:cj.shape[0]/2,:]),2))
#vmin = A.mean()-2.*A.std()
#vmax = A.mean()+A.std()
#print cg.shape,A.shape
gs = plt.GridSpec(2, 4, wspace=0.4, hspace=0.4)
fig = plt.figure(figsize=(6, 6))
fig.add_subplot(gs[:2, :2])
plt.figure(1)
#plt.subplot(121)
x = np.linspace(-25.4, +25.4, A.shape[1], endpoint=True)
y = np.linspace(0, 128, A.shape[0], endpoint=True)
x_ds, y_ds = np.meshgrid( x,y)
A=np.where(A > 0, A, 0.0001)
A=np.log10(np.power(A,2))
print y_ds.shape,x_ds.shape,A.shape
#plt.pcolormesh(x_ds,y_ds, A,  cmap=cm.Greys,vmin=-1,vmax=3)
ax=plt.imshow(np.log10(np.power(A,2)), aspect='auto', cmap=cm.Greys, interpolation='nearest', vmin=-1, origin='lower')
plt.colorbar()
#plt.ylim(0, 128)
#plt.xlim(-25.4,25.4)
plt.ylabel(r"Lag $\tau$ ")
plt.title(r"$|\tilde{E}(\tau,f_D)|$")
plt.xlabel(r"Doppler Frequency $f_D$ ")
#plt.subplot(122)
fig.add_subplot(gs[:2, 2:4])
bina=1
binb=1
a_num=cj.shape[0]
b_num=cj.shape[1]
a_vv = np.copy(cj.reshape(a_num//bina, bina, b_num//binb, binb))
cj=np.mean(np.mean(a_vv,axis=3),axis=1)
cj=np.abs(cj)
cj=np.where(cj > 0, cj, 0.001)
#plt.pcolormesh(x_ds,y_ds,np.log10(cj),  cmap=cm.Greys)
plt.imshow(np.log10(np.power(np.abs(cj),1)), aspect='auto', cmap=cm.Greys, interpolation='nearest', origin='lower')
plt.colorbar()
#plt.ylim(0, 128)
#plt.xlim(-25.4,25.4)
plt.title(r"$|\tilde{I}(\tau,f_D)|$")
#plt.ylabel(r"Lag $\tau$ (ms)")
plt.xlabel(r"Doppler Frequency $f_D$ ")
#plt.subplot(413)
#bina=1
#binb=1
#a_num=dynamic.shape[0]
#b_num=dynamic.shape[1]
plt.figure(2)
#a_vv = np.copy(dynamic.reshape(a_num//bina, bina, b_num//binb, binb))
#dynamic=np.mean(np.mean(a_vv,axis=3),axis=1)
#fig.add_subplot(gs[0:2, :4])
x = np.linspace(0, 8, dynamic.shape[1], endpoint=True)
y = np.linspace(0, 985, dynamic.shape[0], endpoint=True)
x_ds, y_ds = np.meshgrid( x,y)
im = plt.pcolormesh(x_ds, y_ds, dynamic,  cmap=cm.Greys)
#plt.ylim(0, 985)
plt.colorbar()
plt.title("Dynamic Spectrum I(f,t)")
plt.ylabel("Time ")
plt.xlabel("Frequency ")
#fig.add_subplot(gs[2:4, 2:4])
#plt.imshow(np.log10(np.abs(corr)), aspect='auto', cmap=cm.Greys,interpolation='nearest', vmin=-2,origin='lower')
#plt.colorbar()
#plt.title("dynamic spectrum")
#plt.ylabel("time")
#plt.xlabel("frequency")
#plt.subplot(414)
#plt.imshow(a_view, aspect='auto', cmap=cm.Greys, interpolation='nearest', origin='lower')
#plt.colorbar()
#plt.title("rebined dynamic spectrum")
#plt.ylabel("tau")
#plt.xlabel("fd")
plt.show()
