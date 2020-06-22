import numpy as np
import matplotlib.pyplot as plt
import os


def plt_ISI_histogram():
    home = os.path.expanduser('~')

    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    for i in range(3):
        tau_n = [0.1, 1, 3.][i]
        D = 0.1*tau_n
        vR = 0
        vT = 1
        mu = 2.0
        i = 0
        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_{:d}.txt".format(mu, tau_n,
                                                                                                                 D, i)
        print(data_file)
        t_det = np.log((mu - vR) / (mu - vT))
        data = np.loadtxt(data_file)
        t, a, chi, eta, chi2 = np.transpose(data)
        delta_t = [(x - t_det) / t_det for x in t]

        ax.hist(delta_t, bins = 50, density=True, alpha = 0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\langle \delta T \rangle / T$")
    plt.show()
    return 1


if __name__ == "__main__":
    plt_ISI_histogram()