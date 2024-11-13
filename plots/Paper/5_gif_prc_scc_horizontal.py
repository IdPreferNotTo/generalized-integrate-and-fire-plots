import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import styles as st

def gif_prc_theory(t_det, a_det, w_det, gamma, mu, tau_w, beta, tau_a, delta):
    prc = []
    nu = gamma + 1/tau_w
    w = np.sqrt((beta+gamma)/tau_w - nu**2/4)
    for t in np.linspace(0, t_det, 100):
        varprc = np.exp((nu/2)*(t-t_det))*(np.cos(w*(t-t_det)) - ((1.-tau_w*gamma)/(2*tau_w*w))*np.sin(w*(t-t_det)))/(mu-gamma-beta*w_det - a_det + delta/tau_a)
        prc.append(varprc)
    return prc

def gif_prc(t_det, a_det, mu, tau_w, beta, tau_a):
    dt = 10 ** (-6)
    epsilon = 0.001
    ts = np.linspace(0, t_det, 100)
    prc = []
    for tk in ts:
        kick = True
        v = 0
        w = 0
        a = a_det
        t = 0
        while v < 1:
            if (t >= tk and kick == True):
                v = v + epsilon
                kick = False
            v += (mu - v - beta * w - a) * dt
            w += ((v - w) / tau_w) * dt
            a += (-a / tau_a) * dt
            t += dt
        prc.append(-(t - t_det) / epsilon)
    return prc

def chunks(lst, n):
    m = int(len(lst)/n)
    for i in range(0, len(lst), m):
        yield lst[i:i+m]

def plt_scc_gif_theory(gammas, mus, betas, tauws, deltas, tauas, sigmas, Dw):
    home = os.path.expanduser('~')
    sccs_all = []
    sccs_sim_all = []

    prcs = []
    k_min = 1
    k_max = 5
    k_range_theory = range(k_min, k_max + 1)  # np.linspace(k_min, k_max, 30, endpoint=True)
    k_range_sim = range(k_min, k_max + 1)
    t_ratio_range = [0.01, 0.1, 1, 10]
    t_ratioss = [[0.1, 1, 10], [0.1, 1, 10], [0.01, 0.1, 1, 5]]
    t_dets = []

    # Prepare Data
    for n, (gamma, mu, beta, tauw, delta, taua, sigma, t_ratios) in enumerate(zip(gammas, mus, betas, tauws, deltas, tauas, sigmas, t_ratioss)):
        sccs = []
        sccs_sim = []
        t_det, w_det, a_det = fc.get_gif_t_a_w_det(gamma, mu, beta, tauw, taua, delta)
        print(t_det, w_det, a_det)
        t_dets.append(t_det)
        prc = gif_prc_theory(t_det, a_det, w_det, gamma, mu, tauw, beta, taua, delta)
        prcs.append(prc)

        # Calculate SCC for colored noise
        for t_ratio in t_ratios:
            tau_n = t_ratio * t_det
            Dn = sigma * tau_n
            sccs_theory_sub = []
            sccs_sim_sub = []
            data_file = home + "/Data/integrate_and_fire/generalized_if/mu{:.2f}_beta{:.1f}_tauw{:.1f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
                mu, beta, tauw, taua, delta, tau_n, Dn, Dw)
            print(data_file)
            data = np.loadtxt(data_file)
            t, a, eta, chi = np.transpose(data)
            t_mean = np.mean(t)
            delta_t = [x - t_mean for x in t]
            t_det = fc.read_t_det(data_file)

            for k in k_range_theory:
                sccs_theory_sub.append(fc.GIF_scc(t_det, w_det, gamma, mu, tauw, beta, taua, delta, tau_n, Dn, Dw, k))
            for k in k_range_sim:
                c0 = fc.k_corr(delta_t, delta_t, 0)
                ck = fc.k_corr(delta_t, delta_t, k)
                sccs_sim_sub.append(ck / c0)

            delta_ts = list(chunks(delta_t, 25))
            k_corrs_for_var = []
            for ts in delta_ts:
                c0 = fc.k_corr(ts, ts, 0)
                ck = fc.k_corr(ts, ts, 1)
                k_corrs_for_var.append(ck/c0)

            sccs.append(sccs_theory_sub)
            sccs_sim.append(sccs_sim_sub)
        sccs_all.append(sccs)
        sccs_sim_all.append(sccs_sim)

    st.set_default_plot_style()
    colors = st.Colors
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., ratio=0.8))
    x0 = 0.12  # left
    x1 = 0.1  # right
    y0 = 0.12  # bottom
    y1 = 0.10  # ytop
    w_spacer = 0.07
    w_spacer_cbar = 0.05
    h_spacer = 0.15
    width_cbar = 0.025
    width = (1 - 2 * w_spacer - w_spacer_cbar - width_cbar - x0 - x1) / 3
    height = (1 - h_spacer - y0 - y1) / 5

    ax_prc1 = f.add_axes([x0, y0 + 3*height + h_spacer, width, 2*height])
    ax_prc2 = f.add_axes([x0+width+w_spacer, y0 + 3*height + h_spacer, width, 2*height])
    ax_prc3 = f.add_axes([x0+2*width+2*w_spacer, y0 + 3*height + h_spacer, width, 2*height])
    ax_scc1 = f.add_axes([x0, y0, width, 3*height])
    ax_scc2 = f.add_axes([x0+width+w_spacer, y0, width, 3*height])
    ax_scc3 = f.add_axes([x0+2*width+2*w_spacer, y0, width, 3*height])
    ax_cbar1 = f.add_axes([x0+3*width+2*w_spacer+w_spacer_cbar, y0, width_cbar, 3*height])

    axis = [ax_scc1, ax_scc2, ax_scc3, ax_prc1, ax_prc2, ax_prc3, ax_cbar1]
    axis_scc = [ax_scc1, ax_scc2, ax_scc3]
    axis_prc = [ax_prc1, ax_prc2, ax_prc3]
    axis_cbar = [ax_cbar1]

    # Set up plot layout
    ax_prc1.text(-0.1, 1.1, "A$_i$", size=12, transform=ax_prc1.transAxes, ha="center", va="center")
    ax_prc2.text(-0.1, 1.1, "B$_i$", size=12, transform=ax_prc2.transAxes, ha="center", va="center")
    ax_prc3.text(-0.1, 1.1, "C$_i$", size=12, transform=ax_prc3.transAxes, ha="center", va="center")
    ax_scc1.text(-0.1, 1.1, "A$_{ii}$", size=12, transform=ax_scc1.transAxes, ha="center", va="center")
    ax_scc2.text(-0.1, 1.1, "B$_{ii}$", size=12, transform=ax_scc2.transAxes, ha="center", va="center")
    ax_scc3.text(-0.1, 1.1, "C$_{ii}$", size=12, transform=ax_scc3.transAxes, ha="center", va="center")
    #ax_prc1.set_ylim(-1., 6.)
    ax_prc2.set_ylim(-0.15, 1.15)
    #ax_prc1.set_yticks([0, 0.1])
    ax_prc2.set_yticks([0, 1])
    #ax_prc2.set_yticklabels([])
    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        if ax in axis_cbar:
            ax.tick_params(labelsize=utl.labelsize)
        else:
            ax.tick_params(direction='in', labelsize=utl.labelsize)
    for ax in axis_scc:
        ax.set_xticks(k_range_sim)
        ax.set_xlabel("$k$", fontsize=11)
        ax.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    for ax, t_det in zip(axis_prc, t_dets):
        ax.set_xticks([0, t_det / 2, t_det])
        ax.set_xticklabels([0, 0.5, 1])
    for ax in axis_prc:
        ax.set_xlabel(r"$\tau/T^*$", fontsize=11)
    ax_prc1.set_ylabel(r"$Z(\tau)$", fontsize=11)
    ax_scc1.set_ylabel(r"$\rho_k$", fontsize=11)

    ax_prc1.set_title(r"$\nu < 0$", size=11)
    ax_prc2.set_title(r"$ 0 < \nu < 1$", size=11)
    ax_prc3.set_title(r"$\nu > 1$", size=11)

    ax_scc1.set_ylim([-0.7, 0.7])
    ax_scc2.set_ylim([-0.7, 0.7])
    ax_scc1.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax_scc2.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax_scc2.set_yticklabels([])
    # Which parameter should we represent with color?
    colorparams = t_ratio_range
    colormap = cm.viridis
    normalize = mcolors.SymLogNorm(0.01, vmin=np.min(colorparams), vmax=np.max(colorparams))

    # Plot PRCs
    for t_det, prc, ax in zip(t_dets, prcs, axis_prc):
        ax.plot(np.linspace(0, t_det, 100), prc, lw=1, color="k")
        ax.set_xlim([0, t_det])
        prc_neg = [x for x in prc if x < 0]
        prc_pos = [x for x in prc if x >= 0]
        t0_step = len(prc_neg)
        t0 = t0_step/100*t_det
        ax.fill_between(np.linspace(0, t0, t0_step), prc_neg, 0, facecolor=colors.palette[5], alpha=0.5, zorder=2)
        ax.fill_between(np.linspace(t0, t_det, 100-t0_step), prc_pos, 0, facecolor=colors.palette[1], alpha=0.5, zorder=2)

    # Plot SCCs
    for (sccs, sccs_sim, ax, t_ratios) in zip(sccs_all, sccs_sim_all, axis_scc, t_ratioss):
        for n, (scc, scc_sim, ratio) in enumerate(zip(sccs, sccs_sim, t_ratios)):
            color = colormap(normalize(ratio))
            if n == 0:
                ax.plot(k_range_theory, scc, color=color, label="theory",  lw=1, zorder=1)
                ax.scatter(k_range_sim, scc_sim, s=15, color=color, label="sim.",  edgecolors="k", lw = 1, zorder=2)
            else:
                ax.plot(k_range_theory, scc, color=color, lw=1, zorder=1)
                ax.scatter(k_range_sim, scc_sim, s=15, color=color, edgecolors="k", lw = 1, zorder=2)

    # Add legend to last plot
    ax_scc1.legend(fancybox=False, prop={"size": 9}, loc=4, ncol=1, framealpha=1., edgecolor="k")
    leg = ax_scc1.get_legend()
    leg.get_frame().set_linewidth(0.5)
    leg.legendHandles[0].set_color('k')
    leg.legendHandles[1].set_color('k')

    # Plot Colorbars
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    ax_cbar1.set_title(r"$\tau_\eta/T^*$", fontsize=11)
    cbar1 = f.colorbar(s_map, cax=ax_cbar1, format='%.2f')
    cbar1.set_ticks(colorparams)
    ax_cbar1.set_yticklabels([r'$10^{{{:.0f}}}$'.format(np.log(exp)/np.log(10)) for exp in t_ratio_range])


    plt.savefig(home + "/Desktop/bccn_conference_plots/fig5.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    gammas = [1, 1, -1]
    mus = [10, 20, 1] #1.5
    tauas = [10, 10, 1]
    deltas = [10,  10, 2.3]
    betas = [3, 1.5, 5]
    sigmas = [10 ** (-3), 10 ** (-3), 10 ** (-4)]
    Dw = 0
    tauws = [1.5, 1.5, 1.1]
    plt_scc_gif_theory(gammas, mus, betas, tauws, deltas, tauas, sigmas, Dw)
