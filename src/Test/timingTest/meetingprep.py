import numpy as np
import matplotlib.pyplot as plt

SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"

methods = [SEQ, WY1, WY2, YTY1, YTY2, "numpy", "Niliou's seq"]



def load(nc, mc, pc, nprc):
    t_methods = np.zeros((7, 8))
    t_method_err = np.zeros((7, 8))
    counter = 0
    for method in methods:

        n, m, p, thread, npr, t, t_err = np.loadtxt("results/{0}.txt".format(method), unpack=True)
        
        mask = (n==nc) & (m==mc) & (p==pc)  & (npr==nprc)
        t_methods[counter, :] = t[mask]
        t_method_err[counter, :] = t_err[mask]
        counter += 1

    for j in range(len(methods)):
        plt.errorbar(np.arange(1,9), t_methods[j], fmt="-o",yerr=t_method_err[j], label=methods[j])
    #plt.ylim((0, 10))
    plt.title("Execution time with n = {0}, m = {1}, p = {2}".format(nc,mc,pc))
    plt.ylabel("Execution Time (s)")
    plt.xlabel("Number of threads")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()

    return t_methods

plt.figure()
t_methods = load(8, 200, 10, 1)
plt.show()
