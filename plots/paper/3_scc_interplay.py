import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl

def plt_special_cases():
    home = os.path.expanduser("~")
    print(utl.adjust_plotsize(1., 0.5))
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., 0.5))
    x0 = 0.12
    x1 = 0.05
    y0 = 0.2
    y1 = 0.10
    hspacer = 0.05
    wspacer = 0.10
    height = (1 - hspacer - y0 - y1)
    width = (1 - wspacer - x0 - x1) /2
    ax_l = f.add_axes([x0, y0, width, height])
    ax_r = f.add_axes([x0 + wspacer + width, y0, width, height])
    axis = [ax_l, ax_r]

    k_range = range(1, 6)

    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which = "both", direction='in', labelsize=utl.labelsize)
        ax.set_xlim([min(k_range)-0.2, max(k_range)+0.2])
        ax.set_xticks(k_range)
    ax_r.set_yticks([-0.6, -0.3, 0, 0.3, 0.6, 0.9])
    ax_l.set_ylabel(r"$\rho_k$", fontsize=utl.fontsize)
    ax_l.set_xlabel(r"$k$", fontsize=utl.fontsize)
    ax_r.set_xlabel(r"$k$", fontsize=utl.fontsize)


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
        print(t_det)
        mu_cnoise = 1./(1.-np.exp(-t_det))

        data_file1 = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu, tau_a, delta, tau_n, Dn, Dw)
        data_file2 = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu, tau_a, delta, 0, 0, Dw)
        data_file3 = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(mu_cnoise, tau_a, 0, tau_n, Dn, Dw)

        print(data_file1, data_file2, data_file3)
        data1 = np.loadtxt(data_file1)
        t, a, eta, chi = np.transpose(data1)
        t_mean = np.mean(t)
        print(t_mean)
        delta_t1 = [x - t_mean for x in t]

        data2 = np.loadtxt(data_file2)
        t, a, eta, chi = np.transpose(data2)
        t_mean = np.mean(t)
        print(t_mean)
        delta_t2 = [x - t_mean for x in t]

        data3 = np.loadtxt(data_file3)
        t, a, eta, chi = np.transpose(data3)
        t_mean_cnoise = np.mean(t)
        print(t_mean_cnoise)
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

        for k in k_range:
            covar_t1 = fc.k_corr(delta_t1, delta_t1, k)
            covar_t2 = fc.k_corr(delta_t2, delta_t2, k)
            covar_t3 = fc.k_corr(delta_t3, delta_t3, k)
            scc_sim.append(covar_t1 / var_delta_t1)
            scc_adap_sim.append(covar_t2/var_detla_t2)
            scc_cnoise_sim.append(covar_t3 / var_detla_t3)
            print("--------------")
            scc_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, tau_n, Dn, Dw, k))
            print("--------------")
            scc_adap_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, 0, 0, Dw, k))
            scc_cnoise_theory.append(fc.LIF_scc(t_det, mu_cnoise, 0, 0, tau_n, Dn, Dw, k))

        ax.scatter(k_range, scc_sim,  label="sim.", ec="k", fc = "w", zorder=2)
        ax.scatter(k_range, scc_adap_sim, ec="k", fc="C0",  zorder=2)
        ax.scatter(k_range, scc_cnoise_sim, ec="k", fc="C1", zorder=2)
        ax.plot(k_range, scc_theory, label="theory", c="k", ls="-", lw=1, zorder=1)
        ax.plot(k_range, scc_adap_theory, ls="-", lw=1, zorder=1)
        ax.plot(k_range, scc_cnoise_theory, ls="-", lw=1, zorder=1)
        ax.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_l.text(-0.2, 1.1, "(a)", size=10, weight = 'heavy', transform=ax_l.transAxes)
    ax_r.text(-0.2, 1.1, "(b)", size=10, weight = 'heavy', transform=ax_r.transAxes)
    ax_l.text(1.6, -0.28, r"$\rho_{k,a}$", ha="center", fontsize=utl.fontsize, c="C0")
    ax_l.text(3.5, 0.05, r"$\rho_{k,\eta}$", ha="center", fontsize=utl.fontsize, c="C1")
    ax_r.text(3.5, -0.2, r"$\rho_{k,a}$", ha="center", fontsize=utl.fontsize, c="C0")
    ax_r.text(4.5, 0.65, r"$\rho_{k,\eta}$", ha="center", fontsize=utl.fontsize, c="C1")
    ax_l.legend(prop={"size": 7}, framealpha=1., edgecolor="k")
    plt.savefig(home + "/Data/Plots_paper/3_scc_interplay.pdf", transparent=True)
    plt.show()


if __name__ == "__main__":
    plt_special_cases()

