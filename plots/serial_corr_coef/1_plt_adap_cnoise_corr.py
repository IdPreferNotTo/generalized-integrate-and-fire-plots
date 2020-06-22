import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc


def plt_c_noise_corr(mu, tau_a, tau_n, gamma, delta, D):
    home = os.path.expanduser('~')
    data_file = home + "/Data/cLIF/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.2f}_Delta{:.1f}.txt".format(mu, tau_a,tau_n, D, delta)
    data = np.loadtxt(data_file)
    t_det = fc.read_t_det(data_file)

    t, delta_a, chi, eta = np.transpose(data)
    correlations = []
    k_max = 10
    for k in range(k_max):
        correlations.append([k, fc.k_corr(delta_a, chi, k)])

    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    k, k_corr = np.transpose(correlations)
    ax.grid(which = "major", alpha=0.8, linestyle="--")
    #ax.grid(which="minor", alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax.axhline(delta_a_i_cor, ls='--')
    k = np.linspace(0, k_max, 100)
    ax.plot(k, k_corr[0] * np.exp(-k * t_det / tau_n), label="theory")
    ax.scatter(k, k_corr, c='k', label="simulation", s=10)
    ax.set_xlabel("k")
    ax.set_yscale('log')
    ax.set_ylabel(r"$\langle \delta a_i \chi_{i+k} \rangle$")
    ax.set_xlim([0, k_max])
    plt.tight_layout()
    plt.savefig(home + "/Data/cLIF/plots/adap_cnoise_corr_mu{0:.1f}_taun{1:.1f}_taua{2:.1f}_D{3:.1f}.pdf".format(mu, tau_n, tau_a, D), transparent = True)
    plt.show()


if __name__ == "__main__":
    mu = 5
    tau_a = 10
    tau_n = 1
    gamma = 1
    delta = 1
    D = 0.10
    plt_c_noise_corr(mu, tau_a, tau_n, gamma, delta, D)
