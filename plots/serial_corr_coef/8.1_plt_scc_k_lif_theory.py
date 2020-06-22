import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
import matplotlib.ticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plt_scc_lif_theory(tau_a, gamma, mu, delta, D, ratio):
    home = os.path.expanduser('~')

    corr_theory = []
    corr_sim = []
    corr_adap_theory = []
    corr_adap_sim = []
    cvs = []

    k_min = 1
    k_max = 5
    k_range_theory = range(k_min, k_max+1) #np.linspace(k_min, k_max, 30, endpoint=True)
    k_range_sim = range(k_min, k_max+1)
    tau_ns = [0.05, 0.5, 5]

    # Calculate serial correlation coefficient for white noise
    data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
        mu, tau_a, 0.02, D, delta, ratio)
    data = np.loadtxt(data_file, max_rows = 24_000)
    t, a, eta, chi, chi2 = np.transpose(data)
    t_mean = np.mean(t)
    delta_t = [x - t_mean for x in t]

    t_det = fc.read_t_det(data_file)

    print(0)
    for k in k_range_theory:
        corr_adap_theory.append(fc.LIF_scc(t_det, mu, tau_a, delta, 0, k,,, )
    for k in k_range_sim:
        c0 = fc.k_corr(delta_t, delta_t, 0)
        ck = fc.k_corr(delta_t, delta_t, k)
        corr_adap_sim.append(ck / c0)

    # Calculate serial correlation coefficient for colored noise
    for tau_n in tau_ns:
        print(tau_n)
        corr_theory_sub = []
        corr_sim_sub=[]
        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
            mu, tau_a, tau_n, D, delta, ratio)
        data = np.loadtxt(data_file, max_rows=24_000)
        t, a, eta, chi, chi2 = np.transpose(data)
        t_mean = np.mean(t)
        delta_t = [x - t_mean for x in t]
        t_det = fc.read_t_det(data_file)

        for k in k_range_theory:
            corr_theory_sub.append(fc.LIF_scc(t_det, mu, tau_a, delta, tau_n, k,,, )
        for k in k_range_sim:
            c0 =fc.k_corr(delta_t, delta_t, 0)
            ck =fc.k_corr(delta_t, delta_t, k)
            corr_sim_sub.append(ck/c0)

        cvs.append(np.sqrt(c0) / t_mean)
        corr_theory.append(corr_theory_sub)
        corr_sim.append(corr_sim_sub)

    f, (ax, cax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[15,1]}, figsize=(5, 9 / 3))

    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')

    # Which parameter should we represent with color?
    colorparams = tau_ns
    colormap = cm.viridis
    normalize = mcolors.SymLogNorm(0.01, vmin=np.min(colorparams), vmax=np.max(colorparams))

    for i, tau_n in enumerate(tau_ns):
        color = colormap(normalize(tau_n))
        ax.plot(k_range_theory, corr_theory[i], label ="colored", c = color, lw =1, zorder=1)
        ax.scatter(k_range_sim, corr_sim[i], c = color, s=20, edgecolors="k", zorder=2)


    ax.plot(k_range_theory, corr_adap_theory, label ="white", c = "k", ls = "--", zorder=1)
    ax.scatter(k_range_sim, corr_adap_sim, c='w', edgecolor = "k", s=20, zorder=3)
    ax.set_ylabel(r"$\rho_k$")
    ax.set_xlabel("k")
    ax.set_xlim([k_min-0.2, k_max+0.2])

    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # Use this to emphasize the discrete color values
    cbar = f.colorbar(s_map, cax=cax, format='%.2f')
    cbar.set_ticks(colorparams)
    cbar.ax.set_ylabel(r"$\tau_\eta \times 5$")
    cbar.ax.set_yticklabels([r'$10^{{{:.0f}}}$'.format(np.log(tau/5.)/np.log(10)) for tau in tau_ns])
    cbar.ax.yaxis.set_label_position("left")

    cax_cv = cax.twinx()
    ticks = range(len(tau_ns))
    cax_cv.set_ylim(ticks[0], ticks[-1])
    cax_cv.set_yticks(ticks)
    exp = 0
    for i in range(3):
        if (min(cvs)) < 10**(-i):
            exp = i+1

    cax_cv.set_yticklabels(["{:.0f}".format((10**exp)*cv) for cv in cvs])
    cax_cv.set_ylabel(r"$C_V \times 10^{{-{:d}}}$".format(exp))


    #ax.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/LIF/white/plots/scc_theory_sim_mu{:.1f}_taua{:.1f}_delta{:.1f}_D{:.1f}_ratio{:.2f}.pdf".format(mu, tau_a, delta, D, ratio), transparent=True)
    plt.show()

if __name__ == "__main__":
    for i in range(3):
        mu = [20, 20, 5][i]
        tau_a = 2
        gamma = 1
        delta = [10,4.47,1.][i]
        D=0.1
        ratio = 0.0
        plt_scc_lif_theory(tau_a, gamma, mu, delta, D, ratio)

