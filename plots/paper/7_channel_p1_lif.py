import numpy as np
import matplotlib.pyplot as plt
import os
import cmath
from utilites import plot_parameters as utl
from utilites import functions as fc
from shutil import copyfile

def chunks(lst, n):
    m = int(len(lst)/n)
    for i in range(0, len(lst), m):
        yield lst[i:i+m]


def plot_channel_p1_lif(mus, deltas, sigma, Dws):
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

    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which = "both", direction='in', labelsize=utl.labelsize)
        ax.set_xlim([0.08, 120])
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_ylim([-0.7, 1])
        ax.set_yticks([-0.5, 0, 0.5, 1])
        ax.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_l.set_ylabel(r"$\rho_1$", fontsize=utl.fontsize)
    ax_l.set_xlabel(r"$\tau_c/T^*$", fontsize=utl.fontsize)
    ax_r.set_xlabel(r"$\tau_c/T^*$", fontsize=utl.fontsize)
    ax_l.set_title("weak adaptation")
    ax_r.set_title("strong adaptation")

    home = os.path.expanduser("~")
    for n, (mu, delta) in enumerate(zip(mus, deltas)):
        tau_ns = [0.1 * (1.2)**(4*i) for i in range(-4, 11)]
        p1s_sim_all = []
        p1s_theory_all = []
        p1s_adap_theory = []
        p1s_adap_sim = []
        p1s_cnoise_theory = []
        p1s_cnoise_sim = []
        data_x = []
        data_x_cnoise = []

        Ds = Dws[n]
        for D in Ds:
            p1s_sim = []
            p1s_theory = []
            data_x = []
            for tau_n in tau_ns:
                tau_a = tau_n
                Dn = sigma * tau_n
                data_file = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
                    mu, tau_a, delta, tau_n, Dn, D)
                print(data_file)
                data = np.loadtxt(data_file)
                t, a, eta, chi = np.transpose(data)
                t_mean = np.mean(t)
                delta_t = [x - t_mean for x in t]
                t_det = fc.read_t_det(data_file)

                print("T:", t_det, "D:", D)
                c0 = fc.k_corr(delta_t, delta_t, 0)
                c1 = fc.k_corr(delta_t, delta_t, 1)
                p1s_sim.append(c1 / c0)
                p1s_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, tau_n, Dn, D, 1))
                data_x.append(tau_n / t_det)

            p1s_sim_all.append(p1s_sim)
            p1s_theory_all.append(p1s_theory)

        for tau_n in tau_ns:
            tau_a = tau_n
            data_file = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
                mu, tau_a, delta, tau_n, 0, 0.1)
            print(data_file)
            data = np.loadtxt(data_file)
            t, a, eta, chi = np.transpose(data)
            t_mean = np.mean(t)
            delta_t = [x - t_mean for x in t]
            t_det = fc.read_t_det(data_file)

            print("T:", t_det, "adap")
            c0 = fc.k_corr(delta_t, delta_t, 0)
            c1 = fc.k_corr(delta_t, delta_t, 1)
            p1s_adap_sim.append(c1 / c0)

            t_det, a_det = fc.get_lif_t_a_det(mu, tau_a, delta)
            p1s_adap_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, 0, 0, 0.1, 1))

            tau_a =tau_n
            Dn = sigma * tau_n
            data_file = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
                mu, tau_a, 0, tau_n, Dn, 0)
            #copyfile(data_file, home + "/da" + "/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
            #    mu, tau_a, 0, tau_n, Dn, 0))
            print(data_file)
            data = np.loadtxt(data_file)
            t, a, eta, chi = np.transpose(data)
            t_mean = np.mean(t)
            delta_t = [x - t_mean for x in t]
            t_det = fc.read_t_det(data_file)

            print("T:", t_det, "cnoise")
            c0 = fc.k_corr(delta_t, delta_t, 0)
            c1 = fc.k_corr(delta_t, delta_t, 1)
            p1s_cnoise_sim.append(c1 / c0)
            p1s_cnoise_theory.append(fc.LIF_scc(t_det, mu, 0, 0, tau_n, Dn, 0, 1))
            data_x_cnoise.append(tau_n/t_det)

        if n == 0:
            for p1s_sim, p1s_theory, D in zip(p1s_sim_all, p1s_theory_all, Dws[0]):
                ax_l.scatter(data_x, p1s_sim, ec="k", fc="w", s=10, zorder=2)
                ax_l.plot(data_x, p1s_theory, c="k", ls="-", lw=1, zorder=1)
            ax_l.scatter(data_x, p1s_adap_sim, ec="C0", fc="w", s=10, zorder=2)
            ax_l.scatter(data_x_cnoise, p1s_cnoise_sim, ec="C1", fc="w", s=10, zorder=2)
            ax_l.plot(data_x, p1s_adap_theory, ls="--", zorder=1, label = "$\sigma=0$")
            ax_l.plot(data_x_cnoise, p1s_cnoise_theory, ls="-.", zorder=1, label = "$\Delta=0$")

        if n == 1:
            for p1s_sim, p1s_theory, D in zip(p1s_sim_all, p1s_theory_all, Dws[1]):
                ax_r.scatter(data_x, p1s_sim, ec="k", fc="w", s=10, zorder=2)
                ax_r.plot(data_x, p1s_theory, c="k", ls="-", lw=1, zorder=1)
            ax_r.scatter(data_x, p1s_adap_sim, ec="C0", fc="w", s=10, zorder=2)
            ax_r.scatter(data_x_cnoise, p1s_cnoise_sim, ec="C1", fc="w", s=10, zorder=2)
            ax_r.plot(data_x, p1s_adap_theory, ls="--", zorder=1)
            ax_r.plot(data_x_cnoise, p1s_cnoise_theory, ls="-.", zorder=1)


    ax_l.text(50, -0.2, r"$\sigma = 0$", ha="center", fontsize=10, c="C0")
    ax_l.text(0.5, 0.6, r"$\Delta = 0$", ha="center", fontsize=10, c="C1")

    ax_l.text(10, 0.7, r"$D=0$", ha="center", fontsize=10, rotation=30, c="k")
    ax_l.text(20, 0.3, r"$D=0.01}$", ha="center", fontsize=10, rotation=20, c="k")
    ax_l.text(40, 0.1, r"$D=0.05$", ha="center", fontsize=10, rotation=10, c="k")

    ax_r.text(10, 0.7, r"$D=0$", ha="center", fontsize=10, rotation=35, c="k")
    ax_r.text(20, 0.3, r"$D=0.005$", ha="center", fontsize=10, rotation=25, c="k")
    ax_r.text(40, 0.1, r"$D=0.05$", ha="center", fontsize=10, rotation=15, c="k")

    ax_l.text(-0.2, 1.1, "(a)", size=10, weight = 'heavy', transform=ax_l.transAxes)
    ax_r.text(-0.2, 1.1, "(b)", size=10, weight = 'heavy', transform=ax_r.transAxes)
    #ax_l.legend(prop={"size": 7}, loc=7, ncol=1, framealpha=1., edgecolor="k")
    #leg = ax_l.get_legend()
    #leg.get_frame().set_linewidth(0.5)

    plt.savefig(home + "/Data/Plots_paper/7_channel_lif_p1.pdf".format(Dn), transparent=True)
    plt.show()
    return 1

if __name__ == "__main__":
    gamma = 1.
    mus = [5, 20]
    v_t = 1
    deltas = [2, 20]
    sigma =  0.1
    Dws = [[0.00, 0.01, 0.05], [0.00, 0.005, 0.05]]
    plot_channel_p1_lif(mus, deltas, sigma, Dws)