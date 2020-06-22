import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erf
import os

def plt_sta():

    home = os.path.expanduser('~')
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    data_x = []
    data_y = []
    data_y2 = []
    data_file = home + "/Data/LIF/white/data/STA_mu1.10_tau_a0.0_tau_n10.0_D1.00000e-01_Delta0.0_0.txt"
    data = np.loadtxt(data_file, max_rows=2500)

    mu = 1.1
    sigma = 0.1/10.
    f_vT = mu -1.
    c = (f_vT / 2) * erfc(-(f_vT) / np.sqrt(2 * sigma)) + np.power(2 * np.pi * sigma, -1 / 2) * sigma * np.exp(
        -f_vT * f_vT / (2 * sigma))

    data_t  = np.transpose(data)
    for i, x in enumerate(data_t):
        data_x.append(i/100)
        data_y.append(np.mean(x))

    biased_mean = (1/(2*c))*sigma * erfc(-(f_vT)/(sigma*np.sqrt(2)))
    ax.axhline(biased_mean, c="C3")
    ax.tick_params(direction='in')
    ax.set_xlim([0, 1])
    ax.set_ylabel(r"$\langle \eta \rangle $")
    ax.set_xlabel(r"$t/T_i$")
    ax.scatter(data_x, data_y)
    plt.show()

    return 1

if __name__ == "__main__":
    plt_sta()