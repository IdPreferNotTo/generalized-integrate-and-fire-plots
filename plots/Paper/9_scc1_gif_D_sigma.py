import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import styles as st

def plt_special_cases():
    home = os.path.expanduser("~")
    st.set_default_plot_style()
    colors = st.Colors
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., 0.7))
    x0 = 0.12
    x1 = 0.05
    y0 = 0.2
    y1 = 0.10
    hspacer = 0.05
    wspacer = 0.10
    height = (1 - hspacer - y0 - y1) / 5
    width = (1 - wspacer - x0 - x1) / 2

    ax_ld = f.add_axes([x0, y0, width, 3 * height])
    ax_lu = f.add_axes([x0, y0 + 3 * height + hspacer, width, 2 * height])

    ax_rd = f.add_axes([x0 + wspacer + width, y0, width, 3 * height])
    ax_ru = f.add_axes([x0 + wspacer + width, y0 + 3 * height + hspacer, width, 2 * height])

    axins_scc_r = inset_axes(ax_rd, width="100%", height="100%", bbox_to_anchor=(.2, .57, .40, .40),
                             bbox_transform=ax_rd.transAxes)
    axins_scc_l = inset_axes(ax_ld, width="100%", height="100%", bbox_to_anchor=(.6, .57, .40, .40),
                             bbox_transform=ax_ld.transAxes)
    axis = [ax_lu, ax_ld, ax_ru, ax_rd, axins_scc_r, axins_scc_l]
    axins = [axins_scc_r, axins_scc_l]
    axis_lr = [ax_lu, ax_ld, ax_ru, ax_rd]

    axins_scc_l.set_ylim([-0.3, 0.05])
    axins_scc_r.set_ylim([-0.3, 0.3])
    axins_scc_r.set_yticks([-0.2, 0.2])
    ax_lu.set_ylim([0, 1])
    ax_ru.set_ylim([0., 0.5])
    for ax in axins:
        ax.set_ylabel(r"$\rho_k$", fontsize=11)
        ax.set_xticks([1, 5])
        #ax.set_xlabel("$k$", fontsize=11)
    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    for ax in axis_lr:
        ax.set_xscale("log")

    ax_rd.set_ylim([-0.3, 0.5])
    ax_ld.set_ylim([-0.3, 0.5])
    ax_ld.set_xticks([0.001, 0.01, 0.1, 1])
    ax_rd.set_xticks([0.001, 0.01, 0.1, 1])
    ax_lu.set_ylabel(r"$C_v$", fontsize=11)
    ax_ld.set_ylabel(r"$\rho_1$", fontsize=11)
    ax_ld.set_xlabel(r"$D$", fontsize=11)
    ax_rd.set_xlabel(r"$\tau_\eta \sigma^2$", fontsize=11)
    ax_lu.xaxis.set_ticklabels([])
    ax_ru.xaxis.set_ticklabels([])

    gamma = 1
    mu = 20
    tau_w = 1.5
    beta_w = 1.5
    tau_a = 10
    delta = 10
    tau_n = 1
    t_det, w_det, a_det = fc.get_gif_t_a_w_det(gamma, mu, beta_w, tau_w, tau_a, delta)
    print(t_det)
    Dws = [0.001 * 10 ** (k / 5) for k in range(16)]
    Dns = [0.001 * 10 ** (k / 5) for k in range(16)]
    Dws = Dws[::]
    Dns = Dns[::]
    Dnf = 0.1
    Dwf = 0.01
    scc_theory1 = []
    scc_sim1 = []
    cv_sim1 = []
    cv_theory1 = []
    scc_theory2 = []
    scc_sim2 = []
    cv_sim2 = []
    cv_theory2 = []

    for n, Dw in enumerate(Dws):
        data_file = home + "/Data/integrate_and_fire/generalized_if/mu{:.2f}_beta{:.1f}_tauw{:.1f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
            mu, beta_w, tau_w, tau_a, delta, tau_n, Dnf, Dw)
        # print(data_file)
        data = np.loadtxt(data_file)
        t, a, eta, chi = np.transpose(data)
        t_mean = np.mean(t)
        t_std = np.std(t)
        cv_sim1.append(t_std / t_mean)
        cv2 = fc.coefficient_variation(t_det, w_det, gamma, mu, tau_w, beta_w, tau_a, delta, tau_n, Dnf, Dw)
        cv_theory1.append(np.sqrt(cv2))
        delta_t = [x - t_mean for x in t]
        c0 = fc.k_corr(delta_t, delta_t, 0)
        c1 = fc.k_corr(delta_t, delta_t, 1)
        scc_sim1.append(c1 / c0)
        scc_theory1.append(fc.GIF_scc(t_det, w_det, gamma, mu, tau_w, beta_w, tau_a, delta, tau_n, Dnf, Dw, 1))

        if n == 14:
            scck1 = []
            scck1_theory = []
            ks = range(1, 6)
            for k in ks:
                ck = fc.k_corr(delta_t, delta_t, k)
                scck1.append(ck / c0)
                scck1_theory.append(
                    fc.GIF_scc(t_det, w_det, gamma, mu, tau_w, beta_w, tau_a, delta, tau_n, Dnf, Dw, k))

    print("-----")
    for n, Dn in enumerate(Dns):
        data_file = home + "/Data/integrate_and_fire/generalized_if/mu{:.2f}_beta{:.1f}_tauw{:.1f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
            mu, beta_w, tau_w, tau_a, delta, tau_n, Dn, Dwf)
        data = np.loadtxt(data_file)
        t, a, eta, chi = np.transpose(data)
        t_mean = np.mean(t)
        t_std = np.std(t)
        cv_sim2.append(t_std / t_mean)
        cv2 = fc.coefficient_variation(t_det, w_det, gamma, mu, tau_w, beta_w, tau_a, delta, tau_n, Dn, Dwf)
        cv_theory2.append(np.sqrt(cv2))

        delta_t = [x - t_mean for x in t]
        c0 = fc.k_corr(delta_t, delta_t, 0)
        c1 = fc.k_corr(delta_t, delta_t, 1)

        scc_sim2.append(c1 / c0)
        scc_theory2.append(fc.GIF_scc(t_det, w_det, gamma, mu, tau_w, beta_w, tau_a, delta, tau_n, Dn, Dwf, 1))

        if n == 14:
            scck2 = []
            scck2_theory = []
            ks = range(1, 6)
            for k in ks:
                ck = fc.k_corr(delta_t, delta_t, k)
                scck2.append(ck / c0)
                scck2_theory.append(
                    fc.GIF_scc(t_det, w_det, gamma, mu, tau_w, beta_w, tau_a, delta, tau_n, Dn, Dwf, k))

    axins_scc_l.scatter(ks, scck1, label="sim.", ec="k", fc="w", s=10, zorder=2)
    axins_scc_l.plot(ks, scck1_theory, label="theory", c="k", ls="-", lw=1, zorder=1)
    axins_scc_l.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_ld.annotate("",
                   xy=(Dws[14], scc_sim1[14]), xycoords='data',
                   xytext=(Dws[14], 0.1), textcoords='data',
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="arc3,rad=0."),
                   )

    ax_lu.scatter(Dws, cv_sim1, ec="C0", fc="w", s=10, zorder=2, label="sim.")
    ax_lu.plot(Dws, cv_theory1, c="k", ls="-", lw=1, zorder=1, label="theory")
    ax_ld.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_ld.scatter(Dws, scc_sim1, label="sim.", ec="k", fc="w", s=10, zorder=2)
    ax_ld.plot(Dws, scc_theory1, label="theory", c="k", ls="-", lw=1, zorder=1)

    axins_scc_r.scatter(ks, scck2, ec="k", fc="w", s=10, zorder=2)
    axins_scc_r.plot(ks, scck2_theory, c="k", ls="-", lw=1, zorder=1)
    axins_scc_r.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_rd.annotate("",
                   xy=(Dns[14], scc_sim2[14]), xycoords='data',
                   xytext=(Dws[9], 0.4), textcoords='data',
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle,angleA=0,angleB=90,rad=0"),
                   )

    ax_ru.scatter(Dns, cv_sim2, ec="C0", fc="w", s=10, zorder=2)
    ax_ru.plot(Dns, cv_theory2, c="k", ls="-", lw=1, zorder=1)
    ax_rd.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ax_rd.scatter(Dns, scc_sim2, label="sim.", ec="k", fc="w", s=10, zorder=2)
    ax_rd.plot(Dns, scc_theory2, label="theory", c="k", ls="-", lw=1, zorder=1)

    ax_lu.text(-0.2, 1.1, "A", size=12, transform=ax_lu.transAxes)
    ax_ru.text(-0.2, 1.1, "B", size=12, transform=ax_ru.transAxes)
    ax_lu.legend(prop={"size": 7}, framealpha=1., edgecolor="k")
    plt.savefig(home + "/Desktop/bccn_conference_plots/fig9.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plt_special_cases()
