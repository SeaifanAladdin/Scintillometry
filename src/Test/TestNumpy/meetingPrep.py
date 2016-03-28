import numpy as np
import matplotlib.pyplot as plt




def load():
    t = np.zeros((7, 16))
    thread, N, t, t_err = np.loadtxt("NumpyDot.txt", unpack=True)
    
    thread_unique = np.unique(thread)
    N_unique = np.unique(N)

   
    for Ni in N_unique:
        ts = t[np.where(Ni == N)]
        plt.plot(thread_unique, ts, "-o", label=str(Ni))
    plt.title("numpy.dot execution time")
    plt.ylabel("Execution Time (s)")
    plt.xlabel("Number of threads")
    plt.yscale('log')
    
    plt.xscale('log')
    plt.xlim(0,20)
    plt.legend()


plt.figure()
load()
plt.show()
