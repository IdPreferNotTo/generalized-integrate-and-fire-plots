import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from utilites import functions as fc
from utilites import plot_parameters as utl
from matplotlib.collections import LineCollection


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


if __name__ == "__main__":
    home = os.path.expanduser("~")
    data = np.loadtxt(home + "/Data/Taub_miles/taubs_miles_adap_cnoise_I0.0_Dv0.10_Dn1.00.dat")
    v, a, eta, n, m, h, Ia, t = np.transpose(data)

    fig = plt.figure(tight_layout=True, figsize=utl.adjust_plotsize(1., 0.5))
    gs = gs.GridSpec(5, 2)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[2:5, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.text(-0.2, 1.5, "A", size=12, transform=ax1.transAxes)
    ax3.text(-0.2, 1.1, "B", size=12, transform=ax3.transAxes)

    # I = 10.
    ISI = 10.1715
    tau_a = 1 / 0.01
    z_fix = 0.0452
    g = 5
    deltaV = 33
    a_fix = g * deltaV * z_fix
    Delta = tau_a * a_fix * (1. - np.exp(-ISI / tau_a))

    # I = 5.
    ISI = 18.98
    tau_a = 100
    z_fix = 0.0254
    g = 5
    deltaV = 33
    a_fix = g * deltaV * z_fix
    Delta = tau_a * a_fix * (1. - np.exp(-ISI / tau_a))
    print(a_fix)

    ax1.plot(t, a, lw=1, c="k")
    ax1.axhline(z_fix, ls=":", c="k")

    ax1.set_xlim([0, 110])
    ax1.set_xticklabels([])
    ax1.set_ylabel("$z$")
    ax1.set_ylim([0, z_fix * 1.6])
    ax1.set_yticks([0, z_fix])
    ax1.set_yticklabels([0, "$z^*$"])
    ax1.grid(which='major', alpha=0.8, linestyle="--")
    ax1.tick_params(which="both", direction='in', labelsize=utl.labelsize)

    # inset axes....
    axins = ax1.inset_axes([0.05, 0.3, 0.4, 0.6])
    axins.plot(t, a, lw=1, c="k")
    # sub region of the original image
    x1, x2, y1, y2 = 83, 101, 0.016, 0.024
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    ax1.indicate_inset_zoom(axins, edgecolor="k", alpha=1.)

    ax2.plot(t, v, lw=1, c="k")
    ax2.plot([0, 25, 25, 200], [-99, -99, -80, -80], c="C3")

    ax2.set_ylabel("$V$ [mV]")
    ax2.set_ylim([-100, 60])
    ax2.set_yticks([-100, -50, 0, 50])
    ax2.set_xlabel("$t$ [ms]")
    ax2.set_xlim([0, 110])
    ax2.grid(which='major', alpha=0.8, linestyle="--")
    ax2.tick_params(which="both", direction='in', labelsize=utl.labelsize)

    data = np.loadtxt(home + "/Data/Taub_miles/taubs_miles_adap_cnoise_I0.0_Dv0.00_Dn0.00.dat")
    v_det, a_det, eta_det, n, m, h, Ia_det, t_det = np.transpose(data)
    v_det_plot = v_det[900000:920000]
    a_det_plot = a_det[900000:920000]
    ax3.plot(v_det_plot, a_det_plot, ls="--", c="k")

    v_plot = v[850000:870000]
    a_plot = a[850000:870000]
    ax3.plot(v_plot, a_plot, c="C0")
    ax3.grid(which='major', alpha=0.8, linestyle="--")
    ax3.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax3.set_xlabel("$V$ [mV]")
    ax3.set_ylabel("$z$")

    plt.savefig(home + "/Data/Plots/fig10.pdf")
    plt.show()
