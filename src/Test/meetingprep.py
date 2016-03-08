import numpy as np
import matplotlib.pyplot as plt

SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"

methods = [SEQ, WY1, WY2, YTY1, YTY2, "numpy", "Niliou's seq"]



def load(n,m, p):
    t_methods = np.zeros((7, 16))
    t_method_err = np.zeros((7, 16))
    for j in range(1, 16 + 1):
        t, t_err = np.loadtxt("results/testThreading2/threads{0}-n{1}m{2}p{3}.txt".format(j, n, m, p), unpack=True)
        t_methods[:, j - 1] = t
        t_method_err[:, j - 1] = t_err

    th_arr = np.arange(1, len(t) + 1)
    for j in range(len(methods)):
        plt.errorbar(np.arange(1,17), t_methods[j], fmt="-o",yerr=t_method_err[j], label=methods[j])
    #plt.ylim((0, 10))
    plt.title("Execution time with n = {0}, m = {1}, p = {2}".format(n,m,p))
    plt.ylabel("Execution Time (s)")
    plt.xlabel("Number of threads")
    plt.legend()

    return t_methods

plt.figure()
t_methods = load(4, 300, 2)
plt.show()
