import os

import matplotlib.pyplot as plt
import numpy as np

from utilites import functions as fc
from utilites import plot_parameters as utl


def plt_special_cases():
    home = os.path.expanduser("~")
    f = plt.figure(tight_layout = True, figsize=utl.adjust_plotsize(1., 0.5))
    x0 = 0.12
    x1 = 0.05
    y0 = 0.2
    y1 = 0.10
    wspacer = 0.10
    height = (1 - y0 - y1)
    width = (1 - wspacer - x0 - x1) / 2

    ax_l = f.add_axes([x0, y0, width, height])
    ax_r = f.add_axes([x0 + wspacer + width, y0, width, height])
    axis = [ax_l, ax_r]

    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which="both", direction='in', labelsize=utl.labelsize)
        ax.set_xscale("log")


    #ax_l.set_ylim([0.0, 0.2])
    ax_l.set_xticks([0.001, 0.01, 0.1, 1, 10])
    ax_r.set_ylim([0.2, 0.3])
    ax_r.set_xticks([0.001, 0.01, 0.1, 1, 10])

    ax_l.set_ylabel(r"$\langle \delta T_i \delta T_{i+k} \rangle$", fontsize=utl.fontsize)
    ax_l.set_xlabel(r"$\tau_\eta T^*$", fontsize=utl.fontsize)

    ax_r.set_ylabel(r"$\langle T \rangle, T^*$", fontsize=utl.fontsize)
    ax_r.set_xlabel(r"$\tau_\eta / T^*$", fontsize=utl.fontsize)

    mu = 5
    tauns = np.logspace(-2, 1, 20)
    sigma2 = 0.25
    Dw = 0.25
    #Dn = 0.25
    taua = 1
    delta = 0
    t_det, a_det = fc.get_lif_t_a_det(mu, taua, 0)

    covars_sim = []
    vars_sim = []
    covars_the = []
    vars_the = []
    ts_mean = []

    for n, taun in enumerate(tauns):
        Dn = sigma2 * taun
        data_file = home + "/Data/SCC/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}_0.txt".format(
            mu, 0, 0, taun, Dn, Dw)
        #print(data_file)
        data = np.loadtxt(data_file)
        isis, a, eta, chi = np.transpose(data)
        t_mean = np.mean(isis)
        ts_mean.append(t_mean)
        t_std = np.std(isis)
        print(t_std/t_mean)

        beta  = np.exp(-t_det/taun)
        c0 = fc.k_corr(isis, isis, 0)
        c1 = fc.k_corr(isis, isis, 1)
        covars_sim.append(c1)
        vars_sim.append(c0)
        chi1 = 0
        chi0 = 0
        ts = np.linspace(0, t_det, 50)
        dt = t_det / 50.
        sigma2 = Dn/taun
        for t1 in ts:
            Zt1 = fc.lif_varprc(t1, t_det, a_det, mu, 1., 0.)
            chi0 += 2 * Dw  * (Zt1) ** 2 * dt
            for t2 in ts:
                Zt2 = fc.lif_varprc(t2, t_det, a_det, mu, 1., 0.)

                chi1 += sigma2 * Zt1 * Zt2 * np.exp(-(t2 - t1) / taun) * dt * dt
                chi0 += sigma2 * Zt1 * Zt2 * np.exp(-abs(t2 - t1) / taun) * dt * dt

        covars_the.append(chi1 * beta)
        vars_the.append(chi0)

    ax_l.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=2)
    ax_l.scatter([tau/t_det for tau in tauns], covars_sim, label="sim.", ec="k", fc="w", s=10, zorder=2)
    ax_l.plot([tau/t_det for tau in tauns], covars_the, label="theory", color="k", ls="-", lw=1, zorder=1)
    ax_l.scatter([tau/t_det for tau in tauns], vars_sim, ec="C0", fc="w", s=10, zorder=2)
    ax_l.plot([tau/t_det for tau in tauns], vars_the, color="C0", ls="-", lw=1, zorder=1)
    ax_l.text(10, 0.005, r"$k = 0$", ha="center", fontsize=utl.fontsize, c="C0")
    ax_l.text(10, 0.0015, r"$k = 1$", ha="center", fontsize=utl.fontsize, c="k")
    ax_l.legend(loc = 6, prop={"size": 7}, framealpha=1., edgecolor="k")

    ax_r.scatter([tau/t_det for tau in tauns], ts_mean, label=r"$\langle T \rangle$", ec="k", fc="w", s=10, zorder=1)
    ax_r.plot([tau/t_det for tau in tauns], [t_det]*len(tauns), label=r"$T^*$", c="k", ls="-", lw=1, zorder=5)
    ax_r.legend(prop = {"size": 7}, framealpha = 1., edgecolor = "k")

    plt.tight_layout()
    plt.savefig(home + "/Data/SCC/Plots_paper/1_reply.pdf")
    plt.show()

if __name__ == "__main__":
    plt_special_cases()
