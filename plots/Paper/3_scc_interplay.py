import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl
from matplotlib import gridspec

import styles as st

def chunks(lst, n):
    m = int(len(lst)/n)
    for i in range(0, len(lst), m):
        yield lst[i:i+m]

def plt_special_cases():
    home = os.path.expanduser("~")
    print(utl.adjust_plotsize(1., 0.5))
    st.set_default_plot_style()
    colors = st.Colors()
    fig = plt.figure(tight_layout=True, figsize=(6.4*(4/5), 2.4))
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    axis = [ax_l, ax_r]

    k_range = range(1, 6)

    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which = "both", direction='in', labelsize=utl.labelsize)
        ax.set_xlim([min(k_range)-0.2, max(k_range)+0.2])
        ax.set_xticks(k_range)
    ax_r.set_yticks([-0.6, -0.3, 0, 0.3, 0.6, 0.9])
    ax_l.set_ylabel(r"$\rho_k$", fontsize=13)
    ax_r.set_ylabel(r"$\rho_k$", fontsize=13)
    ax_l.set_xlabel(r"$k$", fontsize=13)
    ax_r.set_xlabel(r"$k$", fontsize=13)


    for i, ax in enumerate(axis):
        mu = [5, 20][i]
        tau_a = [2, 1][i]
        tau_n = [0.5, 5][i]  # 0.5, 2
        delta = [2, 10][i]
        sigma = [0.02, 0.02][i]
        Dn = sigma*tau_n
        #Dn = [0.015, 0.1][i]
        Dw = [0.001, 0.001][i]
        t_det, a_det = fc.get_lif_t_a_det(mu, tau_a, delta)
        mu_cnoise = 1./(1.-np.exp(-t_det))

        data_file1 = home + "/Data/integrate_and_fire/leaky_if/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu, tau_a, delta, tau_n, Dn, Dw)
        data_file2 = home + "/Data/integrate_and_fire/leaky_if/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu, tau_a, delta, 0, 0, Dw)
        data_file3 = home + "/Data/integrate_and_fire/leaky_if/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu_cnoise, tau_a, 0, tau_n, Dn, Dw)

        print(data_file1, data_file2, data_file3)
        data1 = np.loadtxt(data_file1)
        t, a, eta, chi = np.transpose(data1)
        t_mean = np.mean(t)
        delta_t1 = [x - t_mean for x in t]

        data2 = np.loadtxt(data_file2)
        t, a, eta, chi = np.transpose(data2)
        t_mean = np.mean(t)
        delta_t2 = [x - t_mean for x in t]

        data3 = np.loadtxt(data_file3)
        t, a, eta, chi = np.transpose(data3)
        t_mean_cnoise = np.mean(t)
        delta_t3 = [x - t_mean_cnoise for x in t]

        scc_sim = []
        scc_theory = []
        scc_adap_theory = []
        scc_adap_sim = []
        scc_cnoise_theory = []
        scc_cnoise_sim = []

        var_delta_t1 = fc.k_corr(delta_t1, delta_t1, 0)
        var_detla_t2 = fc.k_corr(delta_t2, delta_t2, 0)
        var_detla_t3 = fc.k_corr(delta_t3, delta_t3, 0)

        sccs = []
        delta_t1s = fc.chunks(delta_t1, 10)
        for t1 in delta_t1s:
            c0 = fc.k_corr(t1, t1, 0)
            c1 = fc.k_corr(t1, t1, 1)
            sccs.append(c1/c0)
        scc_mean = np.mean(sccs)
        scc_mean_err = np.std(sccs)/np.sqrt(len(sccs)-1)
        print(scc_mean, "error:", scc_mean_err)

        for k in k_range:
            covar_t1 = fc.k_corr(delta_t1, delta_t1, k)
            covar_t2 = fc.k_corr(delta_t2, delta_t2, k)
            covar_t3 = fc.k_corr(delta_t3, delta_t3, k)
            scc_sim.append(covar_t1 / var_delta_t1)
            scc_adap_sim.append(covar_t2/var_detla_t2)
            scc_cnoise_sim.append(covar_t3 / var_detla_t3)

            scc_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, tau_n, Dn, Dw, k))
            scc_adap_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, 0, 0, Dw, k))
            scc_cnoise_theory.append(fc.LIF_scc(t_det, mu_cnoise, 0, 0, tau_n, Dn, Dw, k))

        ax.scatter(k_range, scc_sim,  label="sim.", ec="k", s=20, fc = "w", zorder=2)
        ax.scatter(k_range, scc_adap_sim, ec=colors.palette[3], fc="w",  s=20, zorder=2)
        ax.scatter(k_range, scc_cnoise_sim, ec=colors.palette[5], fc="w", s=20, zorder=2)
        ax.plot(k_range, scc_theory, label="theory", c="k", ls="-", lw=1, zorder=1)
        ax.plot(k_range, scc_adap_theory, ls="-", lw=1, c=colors.palette[3], zorder=1)
        ax.plot(k_range, scc_cnoise_theory, ls="-", lw=1, c=colors.palette[5], zorder=1)
        ax.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_l.text(-0.2, 1.1, "A", size=13, transform=ax_l.transAxes, ha="center", va="center", zorder=5)
    ax_r.text(-0.2, 1.1, "B", size=13, transform=ax_r.transAxes, ha="center", va="center", zorder=5)
    ax_l.text(1.6, -0.25, r"$\rho_{k,a}$", ha="center", fontsize=13, c=colors.palette[3])
    ax_l.text(3.5, 0.08, r"$\rho_{k,\eta}$", ha="center", fontsize=13, c=colors.palette[5])
    ax_r.text(3.5, -0.2, r"$\rho_{k,a}$", ha="center", fontsize=13, c=colors.palette[3])
    ax_r.text(4.5, 0.65, r"$\rho_{k,\eta}$", ha="center", fontsize=13, c=colors.palette[5])
    ax_l.legend(prop={"size": 11}, framealpha=1., edgecolor="k", fancybox=False)
    plt.savefig(home + "/Desktop/bccn_conference_plots/fig3.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plt_special_cases()

