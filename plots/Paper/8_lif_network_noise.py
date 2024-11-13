import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

from utilites import functions as fc
from utilites import plot_parameters as utl
from scipy import integrate
import cmath

import styles as st

def LIF_gnoise_scc(t_det, mu, tau_a, delta, tau_n, bn, bw, k):
    if tau_a == 0:
        alpha = 0
        a_det = 0
        nu = 1

        def pa(i):
            return 0
    else:
        alpha = np.exp(-t_det / tau_a)
        beta = np.exp(-t_det / tau_n) if tau_n != 0 else 0
        a_det = delta / (tau_a * (1 - alpha))

        def h(t):
            return fc.lif_varprc(t, t_det, a_det, mu, tau_a, delta) * np.exp(-t / tau_a)

        zi = integrate.quad(h, 0, t_det)[0]
        nu = 1. - (a_det / tau_a) * zi

        def a(i):
            return alpha * (1 - alpha * alpha * nu) * (1 - nu) * np.real(cmath.exp((i - 1) * cmath.log(alpha * nu)))

        c = 1 + alpha ** 2 - 2 * alpha ** 2 * nu

        def pa(i):
            return -a(i) / c

    if tau_n == 0:
        beta = 0
        chi1 = 0
        chi0 = 1
    else:
        beta = np.exp(-t_det / tau_n)
        chi1 = 0
        chi0 = 0
        ts = np.linspace(0, t_det, 100)
        dt = t_det / 100
        for t1 in ts:
            for t2 in ts:
                Zt1 = fc.lif_varprc(t1, t_det, a_det, mu, tau_a, delta)
                Zt2 = fc.lif_varprc(t2, t_det, a_det, mu, tau_a, delta)
                chi1 += Zt1 * Zt2 * (np.power(bn, 2) / (2 * tau_n) + bn * bw / tau_n) * np.exp(
                    -(t_det + t2 - t1) / tau_n) * dt * dt
                chi0 += Zt1 * Zt2 * (np.power(bn, 2) / (2 * tau_n) + bn * bw / tau_n) * np.exp(
                    -abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            Zt1 = fc.lif_varprc(t1, t_det, a_det, mu, tau_a, delta)
            chi0 += np.power(bw, 2) * Zt1 ** 2 * dt

    def pn(i):
        return beta ** (i - 1) * chi1 / chi0

    A = 1 + (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta) / ((alpha * nu - beta)) * pn(1) - alpha * nu * beta
    B = (1. - (alpha * nu) ** 2) * (1. - alpha * beta) * ((alpha - beta)) / (
            (1 + alpha ** 2 - 2 * alpha ** 2 * nu) * (alpha * nu - beta))
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta
    return (A / C) * pa(k) + (B / C) * pn(k)


def fouriertransformation(w, xs, dt):
    f = 0.
    for i, x in enumerate(xs):
        t = i * dt
        f += x * np.exp(1j * w * t) * dt
    return f


def correlation_function(xs, dt):
    corrs = []
    taus = []
    for i in range(-1000, 1000):
        tau = i * dt
        corr = 0
        if i < 0:
            for x1, x2 in zip(xs[:-i], xs[i:]):
                corr += x1 * x2
            corrs.append(corr)
            taus.append(tau)
        elif i == 0:
            for x1, x2 in zip(xs, xs):
                corr += x1 * x2
            corrs.append(corr)
            taus.append(tau)
        elif i > 0:
            for x1, x2 in zip(xs[i:], xs[:-i]):
                corr += x1 * x2
            corrs.append(corr)
            taus.append(tau)
    return taus, corrs


def slice(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]


def plt_scc_lif(bn, bv):
    home = os.path.expanduser('~')
    st.set_default_plot_style()
    colors = st.Colors
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., ratio=0.8))
    x0 = 0.12  # left
    x1 = 0.1  # right
    y0 = 0.12  # bottom
    y1 = 0.10  # ytop
    w_spacer = 0.12
    w_spacer_cbar = 0.05
    h_spacer = 0.20
    width_cbar = 0.025
    width = (1 - x0 - w_spacer - w_spacer_cbar - width_cbar - x1) / 2
    height = (1 - h_spacer - y0 - y1) / 2
    ax_ps = f.add_axes([x0, y0 + height + h_spacer, width, height])  # top left
    ax_scc_tau = f.add_axes([x0 + width + w_spacer, y0 + height + h_spacer, width, height])  # top right
    ax_no_adap = f.add_axes([x0, y0, width, height])  # bottom left
    ax_weak_adap = f.add_axes([x0 + width + w_spacer, y0, width, height])  # bottom right
    ax_cbar = f.add_axes(
        [x0 + width + width + w_spacer + w_spacer_cbar, y0 + height / 3, width_cbar, 1. - y0 - y1 - 2 * height / 3])

    ax_ps.set_xscale("log")
    ax_ps.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax_ps.set_xlim([0.01, 100])
    ax_ps.set_xticks([0.1, 1, 10.])
    ax_ps.grid(which='major', alpha=0.8, linestyle="--")
    ax_ps.set_ylabel(r"$S_{\zeta\zeta}$", fontsize=11)
    ax_ps.set_xlabel("$\omega T^*$")

    ax_scc_tau.set_xscale("log")
    ax_scc_tau.set_ylabel(r"$\rho_1$", fontsize=11)
    ax_scc_tau.set_xlabel(r"$\tau_\eta / T^*$", fontsize=11)
    ax_scc_tau.grid(which='major', alpha=0.8, linestyle="--")
    ax_scc_tau.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax_scc_tau.set_ylim(-0.65, 0.05)
    ax_scc_tau.set_yticks([-0.6, -0.4, -0.2, 0.0])
    ax_scc_tau.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)

    ax_no_adap.set_xlabel("$k$", fontsize=11)
    ax_no_adap.grid(which='major', alpha=0.8, linestyle="--")
    ax_no_adap.tick_params(direction='in', labelsize=utl.labelsize)
    ax_no_adap.set_xticks([1, 2, 3, 4, 5])
    ax_no_adap.set_ylabel(r"$\rho_k$", fontsize=11)
    ax_no_adap.set_ylim(-0.4, 0.05)
    ax_no_adap.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0])
    ax_no_adap.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)

    ax_weak_adap.grid(which='major', alpha=0.8, linestyle="--")
    ax_weak_adap.tick_params(direction='in', labelsize=utl.labelsize)
    ax_weak_adap.set_xlabel("$k$", fontsize=11)
    ax_weak_adap.set_xticks([1, 2, 3, 4, 5])
    ax_weak_adap.set_ylim(-0.4, 0.05)

    ax_weak_adap.set_ylabel(r"$\rho_k$", fontsize=11)
    ax_weak_adap.set_ylim(-0.4, 0.05)
    ax_weak_adap.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0])
    ax_weak_adap.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)

    ax_ps.text(-0.2, 1.1, "A", size=12, transform=ax_ps.transAxes)
    ax_scc_tau.text(-0.2, 1.1, "B", size=12, transform=ax_scc_tau.transAxes)
    ax_no_adap.text(-0.2, 1.1, "C", size=12, transform=ax_no_adap.transAxes)
    ax_weak_adap.text(-0.2, 1.1, "D", size=12, transform=ax_weak_adap.transAxes)

    t_ratios = [0.1, 1, 10]
    colorparams = t_ratios
    colormap = cm.viridis
    normalize = mcolors.SymLogNorm(0.01, base=10, vmin=np.min(colorparams), vmax=np.max(colorparams))

    # Plot Colorbars
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    ax_cbar.set_title(r"$\tau_\eta / T^*$", fontsize=11)
    cbar1 = f.colorbar(s_map, cax=ax_cbar, format='%.2f')
    cbar1.set_ticks(colorparams)
    ax_cbar.set_yticklabels([r'$10^{{{:.0f}}}$'.format(np.log(exp) / np.log(10)) for exp in t_ratios])

    mu = 5
    delta_no_adap = 0.
    delta_weak_adap = 2.
    taua = 2.
    # PLOT SCC VS. TAU/T^*

    T_no_adap = 0.2231
    T_weak_adap = 0.6652
    ratios = np.logspace(-1, 1, 20)
    taus_no_adap = [r * T_no_adap for r in ratios]
    taus_weak_adap = [r * T_weak_adap for r in ratios]
    tau_correlations_no_adap_sim = []
    tau_correlations_no_adap_the = []
    tau_correlations_weak_adap_sim = []
    tau_correlations_weak_adap_the = []

    for taun_no_adap, taun_weak_adap in zip(taus_no_adap, taus_weak_adap):
        data_no_adap = home + "/Data/integrate_and_fire/leaky_if_network_noise/lif_green_noise_taua{:.2f}_delta{:.2f}_bv{:.2f}_bn{:.2f}_taun{:.2f}.dat".format(
            taua, delta_no_adap, bv, bn, taun_no_adap)
        isis_no_adap = np.loadtxt(data_no_adap)
        covar_no_adap_the = LIF_gnoise_scc(T_no_adap, mu, taua, delta_no_adap, taun_no_adap, bn, bv, 1)
        tau_correlations_no_adap_the.append(covar_no_adap_the)

        var_no_adap_sim = fc.k_corr(isis_no_adap, isis_no_adap, 0)
        covar_no_adap_sim = fc.k_corr(isis_no_adap, isis_no_adap, 1)
        tau_correlations_no_adap_sim.append(covar_no_adap_sim / var_no_adap_sim)

        data_weak_adap = home + "/Data/integrate_and_fire/leaky_if_network_noise//lif_green_noise_taua{:.2f}_delta{:.2f}_bv{:.2f}_bn{:.2f}_taun{:.2f}.dat".format(
            taua, delta_weak_adap, bv, bn, taun_weak_adap)
        isis_weak_adap = np.loadtxt(data_weak_adap)
        covar_weak_adap_the = LIF_gnoise_scc(T_weak_adap, mu, taua, delta_weak_adap, taun_weak_adap, bn, bv, 1)
        tau_correlations_weak_adap_the.append(covar_weak_adap_the)

        var_weak_adap_sim = fc.k_corr(isis_weak_adap, isis_weak_adap, 0)
        covar_weak_adap_sim = fc.k_corr(isis_weak_adap, isis_weak_adap, 1)
        tau_correlations_weak_adap_sim.append(covar_weak_adap_sim / var_weak_adap_sim)

    colors = []
    for r in ratios:
        colors.append(colormap(normalize(r)))
    ax_scc_tau.scatter(ratios, tau_correlations_no_adap_sim, s=15, facecolor=colors, edgecolor="k", lw=1,
                       label="without adaptation", zorder=3.)
    ax_scc_tau.plot(ratios, tau_correlations_no_adap_the, color="k", lw=1., zorder=2.)
    ax_scc_tau.scatter(ratios, tau_correlations_weak_adap_sim, marker="^", s=15, facecolor=colors, edgecolor="k", lw=1,
                       label="with adaptation", zorder=3.)
    ax_scc_tau.plot(ratios, tau_correlations_weak_adap_the, color="k", lw=1., zorder=2.)

    ax_scc_tau.legend(fancybox=False, prop={"size": 7}, loc=4, ncol=1, framealpha=1., edgecolor="k")
    leg_scc_tau = ax_scc_tau.get_legend()
    leg_scc_tau.get_frame().set_linewidth(0.5)
    leg_scc_tau.legendHandles[0].set_color('k')
    leg_scc_tau.legendHandles[1].set_color('k')

    tauns = [0.02231, 0.2231, 2.231]
    for n, taun in enumerate(tauns):
        data_file = home + "/Data/integrate_and_fire/leaky_if_network_noise/lif_green_noise_taua{:.2f}_delta{:.2f}_bv{:.2f}_bn{:.2f}_taun{:.2f}.dat".format(
            taua, delta_no_adap, bv, bn, taun)
        isis = np.loadtxt(data_file)
        t_mean = np.mean(isis)
        CV = np.std(isis) / t_mean
        print(t_mean, CV)

        # SCC FROM SIMULATIONS
        k_correlations = []
        ks = np.arange(1, 6)
        std = fc.k_corr(isis, isis, 0)
        for k in ks:
            rhok = fc.k_corr(isis, isis, k)
            k_correlations.append(rhok / std)

        # SCC FROM THEORY
        k_correlations_theory = []
        for k in ks:
            k_corr = LIF_gnoise_scc(t_mean, mu, taua, delta_no_adap, taun, bn, bv, k)
            k_correlations_theory.append(k_corr)
        color = colormap(normalize(taun / t_mean))
        if n == 0:
            ax_no_adap.scatter(ks, k_correlations, label="sim.", color=color, s=15, edgecolors="k", lw=1, zorder=3)
            ax_no_adap.plot(ks, k_correlations_theory, color=color, label="theory", zorder=2, lw=1)
        else:
            ax_no_adap.scatter(ks, k_correlations, color=color, s=15, edgecolors="k", lw=1, zorder=3)
            ax_no_adap.plot(ks, k_correlations_theory, color=color, zorder=2, lw=1)

        # POWER SPECTRUM FROM THEORY
        ws = np.logspace(-2, 3, 50)

        Sfs = [bv ** 2 + (2 * bn * bv + bn ** 2) / (1. + (taun * w) ** 2) for w in ws]
        ax_ps.plot([w * 0.223 for w in ws], Sfs, c=color, lw=1)

    tauns = [0.06652, 0.6652, 6.652]
    for n, taun in enumerate(tauns):
        data_file = home + "/Data/integrate_and_fire/leaky_if_network_noise/lif_green_noise_taua{:.2f}_delta{:.2f}_bv{:.2f}_bn{:.2f}_taun{:.2f}.dat".format(
            taua, delta_weak_adap, bv, bn, taun)
        isis = np.loadtxt(data_file)
        t_mean = np.mean(isis)
        CV = np.std(isis) / t_mean
        print(t_mean, CV)

        # SCC FROM SIMULATIONS
        k_correlations = []
        ks = np.arange(1, 6)
        std = fc.k_corr(isis, isis, 0)
        for k in ks:
            rhok = fc.k_corr(isis, isis, k)
            k_correlations.append(rhok / std)

        # SCC FROM THEORY
        k_correlations_theory = []
        for k in ks:
            k_corr = LIF_gnoise_scc(t_mean, mu, taua, delta_weak_adap, taun, bn, bv, k)
            k_correlations_theory.append(k_corr)
        color = colormap(normalize(taun / t_mean))
        if n == 0:
            ax_weak_adap.scatter(ks, k_correlations, label="sim.", marker="^", s=15, color=color, edgecolors="k", lw=1,
                                 zorder=3)
            ax_weak_adap.plot(ks, k_correlations_theory, color=color, label="theory", zorder=2, lw=1)
        else:
            ax_weak_adap.scatter(ks, k_correlations, color=color, marker="^", s=15, edgecolors="k", lw=1, zorder=3)
            ax_weak_adap.plot(ks, k_correlations_theory, color=color, zorder=2, lw=1)

    ax_weak_adap.legend(fancybox=False, prop={"size": 7}, loc=4, ncol=1, framealpha=1., edgecolor="k")
    leg = ax_weak_adap.get_legend()
    leg.get_frame().set_linewidth(0.5)
    leg.legendHandles[0].set_color('k')
    leg.legendHandles[1].set_color('k')

    ax_weak_adap.set_title("with adaptation", fontsize=11)
    ax_no_adap.set_title("without adaptation", fontsize=11)

    plt.savefig(home + "/Desktop/bccn_conference_plots/fig8.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    for i in range(1):
        for j in range(1):
            bn = -0.4  # 0.1565 * tau_n  # sigma^2  = D/tau, thus if sigma is to be constant D must be increased propotional to0  tau
            bv = 0.8  # 0.0067
            plt_scc_lif(bn, bv)
