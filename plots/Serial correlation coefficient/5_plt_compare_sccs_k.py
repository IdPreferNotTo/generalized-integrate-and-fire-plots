import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc


def plt_scc(tau_a, tau_n, gamma, mu, delta, v_t, D):
    # Plot scc with adaptation, colored noise and adaptation combined with colored noise
    home = os.path.expanduser('~')
    data_file_adap_cnoise = home + "/Data/LIF/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.2f}_Delta{:.1f}.txt".format(mu, tau_a, tau_n, D, delta)
    data_file_cnoise = home + "/Data/LIF/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.2f}_Delta{:.1f}.txt".format(1.4, 0, tau_n, D, 0)
    data_file_adap = home + "/Data/LIF/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.2f}_Delta{:.1f}.txt".format(mu, tau_a, 0, D, delta)

    t_det_adap_cnoise = fc.read_t_det(data_file_adap_cnoise)
    t_det_cnoise = fc.read_t_det(data_file_cnoise)
    t_det_adap = fc.read_t_det(data_file_adap)

    data_adap_cnoise = np.loadtxt(data_file_adap_cnoise)
    data_cnoise = np.loadtxt(data_file_cnoise)
    data_adap = np.loadtxt(data_file_adap)

    t1, a1, chi1, eta1 = np.transpose(data_adap_cnoise)
    delta_t_adap_cnoise = [x - t_det_adap_cnoise for x in t1]
    t2, a2, chi2, eta2 = np.transpose(data_cnoise)
    delta_t_cnoise = [x - t_det_cnoise for x in t2]
    t3, a3, chi3, eta3 = np.transpose(data_adap)
    delta_t_adap = [x - t_det_adap for x in t3]


    corr_adap_cnoise_theorie = []
    corr_adap_theorie = []
    corr_cnoise_theorie = []
    corr_adap_cnoise_sim = []
    corr_adap_sim = []
    corr_cnoise_sim = []

    variance_adap_cnoise = fc.k_corr(delta_t_adap_cnoise, delta_t_adap_cnoise, 0)
    variance_cnoise = fc.k_corr(delta_t_cnoise, delta_t_cnoise, 0)
    variance_adap = fc.k_corr(delta_t_adap, delta_t_adap, 0)
    k_range = range(1, 5)
    for k in k_range:
        covariance_adap_cnoise = fc.k_corr(delta_t_adap_cnoise, delta_t_adap_cnoise, k)
        covariance_cnoise = fc.k_corr(delta_t_cnoise, delta_t_cnoise, k)
        covariance_adap = fc.k_corr(delta_t_adap, delta_t_adap, k)

        corr_adap_cnoise_sim.append(covariance_adap_cnoise / variance_adap_cnoise)
        corr_cnoise_sim.append(covariance_cnoise / variance_cnoise)
        corr_adap_sim.append(covariance_adap / variance_adap)

        #corr_cnoise_theorie.append(fc.cnoise_scc(chi2, t_det_cnoise, tau_n, k))
        corr_cnoise_theorie.append(fc.cnoise_scc_LIF(t_det_cnoise, tau_n, gamma, mu, D, v_t, k))
        corr_adap_theorie.append(fc.adap_scc(t_det_adap, tau_a, gamma, mu, delta, v_t, k))
        corr_adap_cnoise_theorie.append(fc.adap_cnoise_scc(t_det_adap_cnoise, tau_a, tau_n, delta, gamma, v_t, D, mu, k))

    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')

    ax.plot(k_range, corr_adap_cnoise_theorie, label ="theory", c = "k")
    ax.plot(k_range, corr_adap_theorie, ls = "--", c='k', lw=1)
    ax.plot(k_range, corr_cnoise_theorie, c="k", ls="-.", lw=1)
    ax.scatter(k_range, corr_cnoise_sim, c='k', zorder=3)
    ax.scatter(k_range, corr_adap_sim, c="w", edgecolor='k', zorder=3)
    ax.scatter(k_range, corr_adap_cnoise_sim, label="simulation", c="C0", zorder=3)

    ax.set_ylabel(r"$\rho_k$")
    ax.set_xlabel("k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/cLIF/plots/compare_sccs_case2_mu{:.1f}_taua{:.1f}_delta{:.1f}_D{:.1f}.pdf".format(mu, tau_a, tau_n, delta, D))
    plt.show()

if __name__ == "__main__":
    mu = 80
    tau_a = 10
    tau_n = 0.5
    gamma = 1
    delta = 10
    D = 0.10
    v_t = 1
    D = 0.1
    plt_scc(tau_a, tau_n, gamma, mu, delta, v_t, D)
