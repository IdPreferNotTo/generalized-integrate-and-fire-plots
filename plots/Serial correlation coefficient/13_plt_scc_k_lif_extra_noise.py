import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl

def get_lif_t_a_det(mu, tau_a, delta):
    v = 0
    a = 0
    t = 0
    t_det = 0
    dt = 10**(-5)
    a_tmp0 = -1
    a_tmp1 = 0
    while abs(a_tmp0 - a_tmp1) > 0.001:
        if v < 1.:
            v += (mu - v - a) * dt
            a += (-a / tau_a) * dt
            t += dt
        else:
            t_det = t
            a_tmp1 = a_tmp0
            a_tmp0 = a

            v = 0
            a += delta
            t = 0
    return t_det, a

def plt_scc_lif(tau_a, tau_n, gamma, mu, delta, Dn, Dws):
    home = os.path.expanduser('~')
    corr_adap_cnoise_theory = []
    corr_adap_cnoise_sim = []
    for Dw in Dws:
        data_file = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.2f}_Dn{:.2e}_Dw{:.2e}.txt".format(
            mu, tau_a, delta, tau_n, Dn, Dw)

        print(data_file)
        t_det = fc.read_t_det(data_file)
        data = np.loadtxt(data_file)
        t, a, eta, chi = np.transpose(data)
        t_mean = np.mean(t)
        delta_t = [x - t_mean for x in t]

        corr_adap_cnoise_sim_sub = []
        corr_adap_cnoise_theory_sub = []
        variance_delta_t = fc.k_corr(delta_t, delta_t, 0)
        k_range = range(1, 6)
        for k in k_range:
            covariance_delta_t = fc.k_corr(delta_t, delta_t, k)
            corr_adap_cnoise_sim_sub.append(covariance_delta_t/variance_delta_t)
            corr_adap_cnoise_theory_sub.append(fc.adap_cnoise_extra_noise_LIF_scc(t_det, tau_a, tau_n, gamma, mu, delta, D, Dw, k))
        corr_adap_cnoise_theory.append(corr_adap_cnoise_theory_sub)
        corr_adap_cnoise_sim.append(corr_adap_cnoise_sim_sub)

    f, ax = plt.subplots(1, 1, figsize=utl.adjust_plotsize(0.9))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in', labelsize=utl.labelsize)
    for i, Dw in enumerate(Dws):
        ax.plot(k_range, corr_adap_cnoise_theory[i], ls = "-", lw = 1 )
        ax.scatter(k_range, corr_adap_cnoise_sim[i], label="simulation")
    ax.set_ylabel(r"$\rho_k$", rotation=0, fontsize=utl.fontsize)
    ax.set_xlabel("$k$", fontsize=utl.fontsize)
    #plt.savefig(home + "/Data/LIF/white/plots/scc_full_mu{:.1f}_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.pdf".format(mu, tau_a, tau_n, delta, D))
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        for j in range(1, 2):
                gamma = 1.
                mu = [5, 20][j]
                tau_a = 2
                tau_n = 2.0
                delta = [1, 10][j]
                D = 1*tau_a
                Dws = [0.04*k for k in range(3)]
                utl.set_latex_font()
                plt_scc_lif(tau_a, tau_n, gamma, mu, delta, D, Dws)

