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
    data = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/taubs_miles_adap.dat")
    v, a, n, m, h, t = np.transpose(data)


    fig = plt.figure(tight_layout=True, figsize=utl.adjust_plotsize(1., 0.5))
    gs = gs.GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[0:1, 0])
    ax2 = fig.add_subplot(gs[1:3, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.text(-0.2, 1.5, "(a)", size=10, weight = 'heavy', transform=ax1.transAxes)
    ax3.text(-0.2, 1.1, "(b)", size=10, weight = 'heavy', transform=ax3.transAxes)

    #axins = inset_axes(ax, width="50%", height="40%", loc=4)
    ax1.set_ylabel("$z$")
    ax1.plot(t, a, lw=1, c="k")
    ax1.set_xticklabels([])

    ISI = 10.1715
    tau_a = 1/0.01
    a_fix = 0.0452
    g = 5
    deltaV = 33
    Ia_fix = g * deltaV * 0.0452
    Delta = tau_a * a_fix * (1. - np.exp(-ISI / tau_a))

    ts = np.linspace(0, 11, 1000)
    adaps = [a_fix*np.exp(-(t)/tau_a) for t in ts]
    #ax1.plot(ts, adaps)
    ax1.axhline(a_fix, ls=":", c="k")
    #ax1.axhline(a_fix - Delta/tau_a, ls=":", c="k")
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, a_fix*1.4])
    ax1.grid(which='major', alpha=0.8, linestyle="--")
    ax1.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax1.set_yticks([0, a_fix])
    ax1.set_yticklabels([0, "$z^*$"])

    ax2.set_ylabel("$V$ \ mV")
    ax2.set_xlabel("$t$ \ ms")
    ax2.plot(t, v, lw=1, c="k")
    ax2.plot([0, 50, 50, 200], [-99, -99, -80, -80], c="C3")
    ax2.set_xlim([0, 100])
    ax2.grid(which='major', alpha=0.8, linestyle="--")
    ax2.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax2.set_ylim([-100, 60])
    ax2.set_yticks([-100, -50, 0, 50])

    data = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/taubs_miles_adap_det.dat")
    v_det, a_det, h_det, m_det, n_det, t_det = np.transpose(data)
    dv_det = []
    for v1, v2 in zip(v_det[:-2], v_det[2:]):
        dv_det.append(abs(v2 - v1)*100)

    v_plot = []
    a_plot = []
    c_plot = []
    for v1, a1, v2, a2, c in zip(v_det[50000:51000], a_det[50000:51000], v_det[50001:51001], a_det[50001:51001], dv_det[50000:51000]):
        v_plot.append([v1, v2])
        a_plot.append([a1, a2])
        c_plot.append(c)

    ax3.plot(v_plot, a_plot, ls="--", c="k")
    data = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/taubs_miles_adap_cnoise.dat")
    v, a, eta, Ia, t = np.transpose(data)
    ax3.plot(v[530000:550000], a[530000:550000], c="C0")
    #lc = multiline(v_plot, a_plot, c_plot, cmap="viridis", ax=ax3, norm=matplotlib.colors.LogNorm())
    #axcb = fig.colorbar(lc)
    ax3.grid(which='major', alpha=0.8, linestyle="--")
    ax3.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax3.set_xlabel("$V$ \ mV")
    ax3.set_ylabel("$z$")

    plt.savefig(home + "/Data/Taub_miles/Plots/traub_miles_transient_and_limit_cycle.pdf")
    plt.show()