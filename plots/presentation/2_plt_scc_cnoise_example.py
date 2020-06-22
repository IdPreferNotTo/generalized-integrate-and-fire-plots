import numpy as np
import matplotlib.pyplot as plt
import utilites.functions as fc
import utilites.plot_parameters as utl
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

def plot_scc_cnoise_example():

    f = plt.figure(1, figsize=utl.adjust_plotsize(1.))
    x0 = 0.12
    x1 = 0.10
    y0 = 0.2
    y1 = 0.10
    hspacer = 0.05
    wspacer = 0.05
    width_cbar = 0.05
    height = (1 - hspacer - y0 - y1)
    width = (1 - wspacer - width_cbar - x0 - x1)
    ax = f.add_axes([x0, y0, width, height])
    ax_cbar1 = f.add_axes([x0 + width + wspacer, y0, width_cbar, height])
    k_range = range(1, 6)


    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax.set_xlim([min(k_range) - 0.2, max(k_range) + 0.2])
    ax.set_xticks(k_range)
    ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6, 0.9])
    ax.set_ylabel(r"$\rho_k$", fontsize=utl.fontsize)
    ax.set_xlabel(r"$k$", fontsize=utl.fontsize)
    axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(.6, .6, .4, .4),
                             bbox_transform=ax.transAxes)

    t_det, a_det = fc.get_lif_t_a_det(2, 1, 0)
    prc = []
    ts = np.linspace(0, t_det, 100)
    for t in ts:
        prc.append(fc.lif_varprc(t, t_det, a_det, 2, 1, 0))
    axins.plot(np.linspace(0, t_det, 100), prc, lw=1, c="k")
    axins.set_xticks([0, t_det / 2, t_det])
    axins.set_xticklabels([0, 1 / 2, 1], fontsize=utl.fontsize)
    axins.set_xlabel(r"$\tau/T^*$", fontsize=utl.fontsize)
    axins.set_ylabel(r"$Z(\tau)$", fontsize=utl.fontsize)
    axins.set_xlim([0, t_det])
    axins.set_ylim(0, max(prc) * 1.2)
    axins.fill_between(np.linspace(0, t_det, 100), prc, 0, facecolor="C0", alpha=0.5, zorder=2)



    scc_all = []
    for i in range(1, 5):
        scc = []
        for k in range(1, 6):
            scc.append(fc.LIF_scc(t_det, 2, 0, 0, (i/2)*t_det, 0.1, 0, k))
        scc_all.append(scc)

    colorparams = [0.5, 1, 1.5, 2]
    colormap = cm.viridis
    normalize = mpl.colors.Normalize(vmin=np.min(colorparams),vmax=np.max(colorparams)) #mcolors.SymLogNorm(0.01, vmin=np.min(colorparams), vmax=np.max(colorparams))

    for cpar, scc in zip(colorparams, scc_all):
        color = colormap(normalize(cpar))
        ax.plot([1,2,3,4,5], scc, c=color, zorder=1)
        ax.scatter([1,2,3,4,5], scc, c=color, edgecolors="k", zorder=2)

    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    ax_cbar1.set_title(r"$\tau_\eta/T^*$", fontsize=utl.fontsize)
    cbar1 = f.colorbar(s_map, cax=ax_cbar1, format='%.2f')
    cbar1.set_ticks(colorparams)
    ax_cbar1.set_yticklabels([0.5, 1, 1.5, 2])
    home = os.path.expanduser('~')
    plt.savefig(home + "/Data/Plots_paper/scc_cnoise1.pdf", transparent=True)
    plt.show()
    return 1

if __name__ == "__main__":
    plot_scc_cnoise_example()