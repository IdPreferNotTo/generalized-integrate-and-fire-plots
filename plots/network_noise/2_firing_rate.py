import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    home = os.path.expanduser("~")
    data = np.loadtxt(home + "/CLionProjects/PhD/lif_green_noise/out/lif_green_noise_bv2.32_bn-107.00_an91.43.dat")
    idxs = range(len(data))
    plt.scatter(idxs, data)
    plt.show()