import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl

def plt_scc_lif(taua, taun, mu, delta, Dn, Dw):
    home = os.path.expanduser('~')

    data_file = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu, taua, delta, taun, Dn, Dw)

    print(data_file)
    t_det = fc.read_t_det(data_file)
    data = np.loadtxt(data_file)
    t, a, eta, chi = np.transpose(data)
    t_mean = np.mean(t)
    t_det, a_det = fc.get_lif_t_a_det(mu, tau_a, delta)
    delta_t = [x - t_mean for x in t]
    print(t_det, a_det)
    corr_adap_cnoise_sim = []
    corr_adap_cnoise_theory = []

    # CALCULATE SPEARMAN'S RANK-ORDER CORRELATION
    con_index_ts = []
    for i, x in enumerate(t):
        con_index_ts.append([i, x])
    con_index_ts.sort(key=lambda l: l[1])
    con_index_ts_rank = []
    for r, (i, x) in enumerate(con_index_ts):
        con_index_ts_rank.append([i, x, r+1])
    con_index_ts_rank.sort(key=lambda l: l[0])
    mean_rank = (t.size + 1) / 2
    dif_ranks = [l[2]-mean_rank for l in con_index_ts_rank]
    var = 0
    for dr in dif_ranks:
        var += dr*dr
    var /= len(dif_ranks)
    covars = []
    for k in range(1, 7):
        covar = 0
        for dr1, dr2 in zip(dif_ranks[:-k], dif_ranks[k:]):
            covar += dr1*dr2
        covar /= len(dif_ranks)-1
        covars.append(covar/var)


    variance_delta_t = fc.k_corr(delta_t, delta_t, 0)
    k_range = range(1, 7)
    for k in k_range:
        covariance_delta_t = fc.k_corr(delta_t, delta_t, k)
        corr_adap_cnoise_sim.append(covariance_delta_t/variance_delta_t)
        corr_adap_cnoise_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, tau_n, Dn, Dw, k))


    f, ax = plt.subplots(1, 1, figsize=utl.adjust_plotsize(0.9, 0.5))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in', labelsize=utl.labelsize)
    ax.scatter(k_range, corr_adap_cnoise_sim, label = "Simulation")
    ax.scatter(k_range, covars, label = "Spearman")
    ax.plot(k_range, corr_adap_cnoise_theory, label ="Theory", c = "k", ls = "-", lw = 1 )
    ax.set_ylabel(r"$\rho_k$", rotation=0, fontsize=utl.fontsize)
    ax.set_xlabel("$k$", fontsize=utl.fontsize)
    ax.legend()
    plt.tight_layout()
    #plt.savefig(home + "/Data/LIF/white/plots/scc_full_mu{:.1f}_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.pdf".format(mu, tau_a, tau_n, delta, D))
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        for j in range(1):
            gamma = 1
            mu = 5
            tau_a = 2
            tau_n = 0.5 #0
            delta_a = 2
            Dn = 0.05/tau_n #0
            Dw = 0.001
            plt_scc_lif(tau_a, tau_n, mu, delta_a, Dn, Dw)

