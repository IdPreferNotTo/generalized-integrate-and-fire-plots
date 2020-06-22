import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy import integrate
import os

def theta_model_prc(i):

    home = os.path.expanduser('~')
    mu = [1.01, 1.1, 1.5, 2.][i]
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    data_x = []
    data_y = []
    data_y2 = []
    for i in range(25):
        tau_n = 0

        D=0.02*i
        vR = 0
        vT = 1
        f_vT = mu-vT
        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5e}_Delta0.0_{:d}.txt".format(mu, tau_n, D, 0)
        data = np.loadtxt(data_file, max_rows=25_000)
        t_det = np.log((mu-vR)/(mu-vT))

        t, a, eta, chi, chi2 = np.transpose(data)
        delta_t = [(x - t_det) for x in t]

        data_x.append(D)
        data_y.append(np.mean(delta_t))
        data_y2.append(-D*np.exp(-t_det)/(f_vT*f_vT))

    ax.scatter(data_x, data_y)
    ax.plot(data_x, data_y2, c="C3")
    ax.tick_params(direction='in')
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$\langle \delta T / T\rangle $")
    ax.set_ylim([min(data_y)-0.1, 0.05])
    plt.tight_layout()
    plt.savefig(home + "/Data/LIF/white/plots/LIF_white_noise_mu{:.2f}.pdf".format(mu))
    plt.show()
    return 1

if __name__ == "__main__":
    for i in range(4):
        theta_model_prc(i)