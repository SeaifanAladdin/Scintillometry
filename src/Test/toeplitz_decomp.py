import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import inv, toeplitz
import matplotlib.pylab as plt
import matplotlib.cm as cm

#Schur's algoritm for decomposing a Toeplitz matrix
#Input a is the first column of the Toeplitz matrix to be decomposed
def toeplitz_decomp(a):
    n = len(a)
    l = np.zeros(shape=(n,n), dtype=complex)
    alpha=np.append(-np.conj(a[1:]),0)
    beta=np.conj(a)
    for i in xrange(n):
        if np.real(beta[0]) < 0:
                print("Loop: ",i)
                print("beta[0] = ",np.real(beta[0]))
                print("ERROR - not positive definite")
                break
        s = np.sqrt(np.real(beta[0]))
        l[i:n,i] = np.conj(beta)[0:n-i] / s
        gamma = alpha[0] / np.real(beta[0])
        beta0 = np.array(beta)
        beta -= np.conj(gamma)*alpha
        alpha -= gamma*beta0
        alpha = np.append(alpha[1:],0)
    return l
def toeplitz_blockschur(a,b,pad):
    n=a.shape[1]/b
    g1=np.zeros(shape=(b,n*b), dtype=complex)
    g2=np.zeros(shape=(b,n*b), dtype=complex)
    l=np.zeros(shape=((n+n*pad)*b,(n+n*pad)*b), dtype=complex)
    #for simulated data that the first matrix is toeplitz
    #c=toeplitz_decomp(np.array(a[0:b,0]).reshape(-1,).tolist())
    c=np.linalg.cholesky(a[0:b,0:b])
    for j in xrange(0,n): 
        g2[:,j*b:(j+1)*b]= -np.dot(inv(c),a[0:b,j*b:(j+1)*b]) 
        g1[:,j*b:(j+1)*b]= -g2[:,j*b:(j+1)*b]
        l[0:b,0:n*b] = g1
        
    for i in xrange( 1,n + n*pad):
        start_g1=0
        end_g1=min(n*b,(n+n*pad-i)*b)
        start_g2=i*b
        if (i<n*pad+1):
            g2=np.append(g2,np.zeros(shape=(b,b)),axis=1)
        end_g2=end_g1+i*b
        for j in xrange(0,b):
            sigma=np.dot(np.conj(g2[:,start_g2+j].T),g2[:,start_g2+j])
            alpha=-np.sign(g1[j,start_g1+j])*np.sqrt(g1[j,start_g1+j]**2 - sigma)
            z=g1[j,start_g1+j]+alpha
            if np.count_nonzero(g2[:,start_g2+j])==0 or z==0:
                beta=0
                g1[j,start_g1+j]=-alpha
                g2[:,start_g2+j+1:end_g2]=-g2[:,start_g2+j+1:end_g2]
                continue
            else:
                x2=-np.copy(g2[:,start_g2+j])/np.conj(z)
                beta=(2*z*np.conj(z))/(np.conj(z)*z-sigma)  
                g2[:,start_g2+j]=0
                g1[j,start_g1+j]=-alpha
                v=np.copy(g1[j,start_g1+j+1:end_g1]+np.dot(np.conj(x2.T),g2[:,start_g2+j+1:end_g2]))
                g1[j,start_g1+j+1:end_g1]=g1[j,start_g1+j+1:end_g1]-beta*v
                v=np.reshape(v,(1,v.shape[0]))
                x2=np.reshape(x2,(1,x2.shape[0]))
                g2[:,start_g2+j+1:end_g2]=-g2[:,start_g2+j+1:end_g2]-beta*np.dot(x2.T,v)
        c=min(n+i,n+n*pad)
        l[i*b: (i+1)*b,i*b:c*b]=g1[:,0:c*b-(i*b)]
    return l
def myBlockChol(a,m):
    n=len(a)/m
    l = np.zeros(shape=(m*n,m*n), dtype=complex) 
    l=np.copy(a);
    for j in xrange(0,n):
        s=j*m
        e=j*m+m
        l[s:e,s:e]=toeplitz_decomp(np.array(l[s:e,s]).reshape(-1,).tolist())    
        if j<n-1:
            rs=j*m+m
            re=n*m
        else:
            break
        l[rs:re,s:e]=np.dot(l[rs:re,s:e],inv(np.conj(l[s:e,s:e].T)))
        l[rs:re,rs:re]=l[rs:re,rs:re]- np.dot(l[rs:re,s:e],np.conj(l[rs:re,s:e].T))         
    return np.tril(l)
def toep_zpad(a,npad):
    n = len(a)
    l = np.zeros(shape=(n+npad*n,n+npad*n), dtype=complex)
    alpha=np.append(-np.conj(a[1:]),0)
    beta=np.conj(a)
#    for i in xrange(npad*n):
#        if np.real(beta[0]) < 0:
#            print("Loop: ",i)
#            print("beta[0] = ",np.real(beta[0]))
#            print("ERROR - not positive definite")
#            break

#        gamma = alpha[0] / np.real(beta[0])
#        beta0 = np.array(beta)
#        beta -= np.conj(gamma)*alpha
#        alpha -= gamma*beta0
#        alpha = np.append(alpha[1:],0)
    for i in xrange(n+npad*n):
        if np.real(beta[0]) < 0:
                print("Loop: ",i)
                print("beta[0] = ",np.real(beta[0]))
                print("ERROR - not positive definite")
                break
        s = np.sqrt(np.real(beta[0]))
        c=min(n+npad*n,i+n)
        l[i:c,i] = np.conj(beta)[0:c-i] / s
        gamma = alpha[0] / np.real(beta[0])
        beta0 = np.array(beta)
        beta -= np.conj(gamma)*alpha
        alpha -= gamma*beta0
        alpha = np.append(alpha[1:],0)
    return l
#Schur's algoritm for decomposing a block Toeplitz matrix
#Input a is the first column of the block Toeplitz matrix to be decomposed
def plot_results(A):
    A = np.abs(A)
    vmin = A.mean()-2.*A.std()
    vmax = A.mean()+5.*A.std()

    plt.imshow(A, aspect='auto', cmap=cm.Greys, interpolation='nearest', vmax=vmax, vmin=0, origin='lower')

    plt.colorbar()
    plt.ylabel("tau")
    plt.xlabel("fd")

    plt.show()
