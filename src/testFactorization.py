import matplotlib.pyplot as plt
import numpy as np
import os,sys
from scipy.linalg import cholesky


##Specifications
n=4
m=8

offsetn = 0
offsetm = 140

pad = 1

##Load the processed data 
filename = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n,m*4, offsetn, offsetm) 
file = filename + ".dat"
folder = "processedData"


##Load the toepletz file generated from extract_real
toeplitz1 = np.memmap("{0}/{1}".format(folder,file), dtype='complex', mode='r', shape=(4*m,4*n*m), order='F')

##Creating a square toeplitz matrix
toeplitz = np.zeros((4*m*n*(1 + pad), 4*n*m*(1 + pad)), complex)


##Gonna be messy, but using toeplitz1 to create toeplitz
toeplitz[:4*m, :4*n*m] = toeplitz1

for i in range(0,n*(1 + pad)):
    T_temp = toeplitz[:4*m,4*i*m:4*(i+1)*m]
    for j in range(0,n*(1 + pad) - i):
        k = j + i
        toeplitz[4*j*m:4*(j+1)*m,4*k*m:4*(k+1)*m] = T_temp
        toeplitz[4*k*m:4*(k+1)*m, 4*j*m:4*(j+1)*m] = np.conj(T_temp.T)

##Plotting thr raw toeplitz matrix
fig = plt.figure()
plt.imshow(np.abs(toeplitz))
plt.colorbar()
if not pad:
        plt.title("Toeplitz matrix with n = {0}, m = {1}".format(n, m*4))
else:
        plt.title("Zero padded Toeplitz matrix with n = {0}, m = {1}".format(n, m*4))
##Adding graphing lines
ax = fig.gca()
ax.set_xticks(np.arange(0,4*n*m*(1 + pad),4*m))
ax.set_yticks(np.arange(0,4*n*m*(1 + pad),4*m))
ax.grid(True, which='both')
if pad:
        plt.plot([0, 2*4*n*m], [4*n*m, 4*m*n], '-k')
        plt.plot([4*n*m, 4*m*n],[0, 2*4*n*m], '-k')
plt.show()



##Loading the factorized matrix 
folder = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n, m*4, offsetn, offsetm)


L = np.zeros((4*m*n*(1 + pad),4*m*n*(1 + pad)), complex)
for i in range(n*(1 + pad)):
    for j in range(n*(1 + pad)): 
        path = "results/{0}/L_{1}-{2}.npy".format(folder, i,j)     
        
        if os.path.isfile(path):
            Ltemp = np.load(path)
            L[4*m*j: 4*m*(j + 1), 4*m*i:4*m*(i + 1)] = Ltemp



##The  factorized matrix using Numpy's Cholesky
#npL = cholesky(toeplitz, True)



fig = plt.figure()
plt.subplot(1,2,1)

##Graphical lines
ax = fig.gca()
ax.set_xticks(np.arange(0,4*n*m*(1 + pad),4*m))
ax.set_yticks(np.arange(0,4*n*m*(1 + pad),4*m))
ax.grid(True, which='both')
if pad:
    plt.plot([0, 2*4*n*m], [4*n*m, 4*m*n], '-k')
    plt.plot([4*n*m, 4*m*n],[0, 2*4*n*m], '-k')

##Plotting the error difference between our raw and our results multiplied by its transposed conjugate
plt.imshow(np.abs(toeplitz - L.dot(np.conj(L.T))))
plt.colorbar()
plt.title("Errors on the Toeplitz Matrix")


plt.subplot(1,2,2)
##Graphical lines
ax = fig.gca()
ax.set_xticks(np.arange(0,4*n*m*(1 + pad),4*m))
ax.set_yticks(np.arange(0,4*n*m*(1 + pad),4*m))
ax.grid(True, which='both')
if pad:
        plt.plot([0, 2*4*n*m], [4*n*m, 4*m*n], '-k')
        plt.plot([4*n*m, 4*m*n],[0, 2*4*n*m], '-k')

##Plotting the factorized matrix
plt.title("Factorized Toeplitz Matrix")
plt.imshow(np.abs(L))
plt.colorbar()
plt.show()

