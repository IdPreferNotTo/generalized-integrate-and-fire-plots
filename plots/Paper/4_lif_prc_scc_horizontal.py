import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import styles as st

def lif_prc(t_det, a_det, mu, tau_a):
    dt = 10**(-5)
    epsilon = 0.01
    ts=np.linspace(0, t_det, 100)
    prc = []
    for tk in ts:
        kick=True
        v = 0
        a = a_det
        t = 0
        while v < 1:
            if (t>=tk and kick==True):
                v = v+epsilon
                kick=False
            v += (mu - v - a)*dt
            a += (-a/tau_a)*dt
            t += dt
        prc.append(-(t - t_det)/epsilon)
    return prc

def lif_prc_theory(t_det, a_det, mu, tau_a, delta):
    prc = []
    for t in np.linspace(0, t_det, 100):
        varprc = np.exp(t - t_det)/(mu-1.-a_det+delta/tau_a)
        prc.append(varprc)
    return prc

def plt_scc_lif_theory(mus, tau_a, delta_as, sigma, Dw):
    home = os.path.expanduser('~')
    sccs_all = []
    sccs_sim_all = []
    prcs = []

    k_min = 1
    k_max = 5
    k_range_theory = range(k_min, k_max+1) #np.linspace(k_min, k_max, 30, endpoint=True)
    k_range_sim = range(k_min, k_max+1)
    t_ratios = [0.1, 1, 10]
    t_dets = []

    # Prepare Data
    for n, (mu, delta) in enumerate(zip(mus, delta_as)):
        sccs = []
        sccs_sim = []
        t_det, a_det  = fc.get_lif_t_a_det(mu, tau_a, delta)
        print (t_det, a_det, delta/(tau_a*(1 - np.exp(-t_det/tau_a))))
        t_dets.append(t_det)

        print("T^*: ", t_det)
        prc = lif_prc_theory(t_det, a_det, mu, tau_a, delta)
        prcs.append(prc)

        # Calculate SCC for colored noise
        for t_ratio in t_ratios:
            tau_n = t_ratio*t_det
            Dn = sigma*tau_n
            data_file = home + "/Data/integrate_and_fire/leaky_if/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
                mu, tau_a, delta, tau_n, Dn, Dw)
            print(data_file)
            data = np.loadtxt(data_file)
            t, a, eta, chi = np.transpose(data)
            t_mean = np.mean(t)
            delta_t = [x - t_mean for x in t]

            sccs_theory_sub = []
            sccs_sim_sub=[]
            for k in k_range_theory:
                sccs_theory_sub.append(fc.LIF_scc(t_det, mu, tau_a, delta, tau_n, Dn, Dw, k))
            for k in k_range_sim:
                c0 =fc.k_corr(delta_t, delta_t, 0)
                ck =fc.k_corr(delta_t, delta_t, k)
                sccs_sim_sub.append(ck/c0)
            sccs.append(sccs_theory_sub)
            sccs_sim.append(sccs_sim_sub)
        sccs_all.append(sccs)
        sccs_sim_all.append(sccs_sim)

    # Set up plot positions
    colors = st.Colors
    st.set_default_plot_style()
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., ratio=0.8))
    x0 = 0.12  # left
    x1 = 0.1  # right
    y0 = 0.12  # bottom
    y1 = 0.10  # ytop
    w_spacer = 0.09
    w_spacer_cbar = 0.05
    h_spacer = 0.15
    width_cbar = 0.025
    width = (1 -x0 - w_spacer - w_spacer_cbar - width_cbar - x1)/2
    height = (1 - h_spacer - y0 - y1) / 5
    ax_prc2 = f.add_axes([x0, y0 + 3*height + h_spacer, width, 2*height]) #top left
    ax_prc1 = f.add_axes([x0 + width + w_spacer, y0 + 3* height + h_spacer, width, 2*height]) # top right
    ax_scc2 = f.add_axes([x0, y0, width, 3*height]) # bottom left
    ax_scc1 = f.add_axes([x0+width+w_spacer, y0, width, 3*height]) # bottom right
    ax_cbar = f.add_axes([x0+width+width+w_spacer+w_spacer_cbar, y0, width_cbar, 3*height])
    axis = [ax_scc1, ax_scc2, ax_prc1, ax_prc2, ax_cbar]
    axis_scc = [ax_scc1, ax_scc2]
    axis_prc = [ax_prc1, ax_prc2]

    # Set up plot layout
    ax_prc2.text(-0.1, 1.1, "A$_i$", size=12, transform=ax_prc2.transAxes, ha="center", va="center")
    ax_prc1.text(-0.1, 1.1, "B$_i$", size=12, transform=ax_prc1.transAxes, ha="center", va="center")
    ax_prc2.set_title(r"$\nu < 0$", size=11)
    ax_prc1.set_title(r"$\nu > 0$", size=11)
    ax_scc2.text(-0.1, 1.1, "A$_{ii}$", size=12, transform=ax_scc2.transAxes, ha="center", va="center")
    ax_scc1.text(-0.1, 1.1, "B$_{ii}$", size=12, transform=ax_scc1.transAxes, ha="center", va="center")

    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        if ax==ax_cbar:
            ax.tick_params(labelsize=utl.labelsize)
        else:
            ax.tick_params(direction='in', labelsize=utl.labelsize)
    for ax in axis_scc:
        ax.set_xticks(k_range_sim)
        ax.set_ylim(-0.9, 0.9)
        ax.set_xlabel("$k$", fontsize=11)
        ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
        ax.axhline(0, xmin=0, xmax=6, ls = "--", color="C7", zorder=1)
    ax_scc1.set_yticklabels([])
    for ax, t_det in zip(axis_prc, t_dets):
        ax.set_xticks([0, t_det/2 , t_det])
        ax.set_xticklabels(["0", "0.5" , "1"])
        ax.set_ylim([0, 0.7])
    for ax in axis_prc:
        ax.set_xlabel(r"$\tau/T^*$", fontsize=11)
    ax_prc1.set_yticklabels([])
    ax_prc2.set_ylabel(r"$Z(\tau)$", fontsize=11)
    ax_scc2.set_ylabel(r"$\rho_k$", fontsize=11)
    # Which parameter should we represent with color?
    colorparams = t_ratios
    colormap = cm.viridis
    normalize = mcolors.SymLogNorm(0.01, vmin=np.min(colorparams), vmax=np.max(colorparams))

    # Plot PRCs
    for t_det, prc, ax in zip(t_dets, prcs, axis_prc):
        ax.plot(np.linspace(0, t_det, 100), prc, lw=1, color="k")
        ax.set_xlim([0, t_det])
        ax.fill_between(np.linspace(0, t_det, 100), prc, 0, facecolor=colors.palette[1], alpha=0.5, zorder=2)

    # Plot SCCs
    for (sccs, sccs_sim, ax) in zip(sccs_all, sccs_sim_all, axis_scc):
        for n, (scc, scc_sim, ratio) in enumerate(zip(sccs, sccs_sim, t_ratios)):
            color = colormap(normalize(ratio))
            if n == 0:
                ax.plot(k_range_theory, scc, label ="theory", c = color, lw = 1., zorder=1)
                ax.scatter(k_range_sim, scc_sim, s=15, label="sim.", color = color, edgecolors="k", lw = 1, zorder=2)
            else:
                ax.plot(k_range_theory, scc, color=color, lw=1, zorder=1)
                ax.scatter(k_range_sim, scc_sim, s=15, color = color, edgecolors="k", lw = 1, zorder=2)

    ax_scc2.legend(fancybox = False, prop={"size": 11}, loc=4, ncol=1, framealpha=1., edgecolor="k")
    leg = ax_scc2.get_legend()
    leg.get_frame().set_linewidth(0.5)
    leg.legendHandles[0].set_color('k')
    leg.legendHandles[1].set_color('k')

    # Plot Colorbars
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    ax_cbar.set_title(r"$\tau_\eta / T^*$", fontsize=11)
    cbar1 = f.colorbar(s_map, cax=ax_cbar, format='%.2f')
    cbar1.set_ticks(colorparams)
    ax_cbar.set_yticklabels([r'$10^{{{:.0f}}}$'.format(np.log(exp)/np.log(10)) for exp in t_ratios])

    #ax.legend()
    plt.savefig(home + "/Desktop/bccn_conference_plots/fig4.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    gamma = 1
    mus = [5,20]
    tau_a = 2
    delta_as = [2, 20.]
    sigma=0.1
    Dw=0
    plt_scc_lif_theory(mus, tau_a, delta_as, sigma, Dw)

