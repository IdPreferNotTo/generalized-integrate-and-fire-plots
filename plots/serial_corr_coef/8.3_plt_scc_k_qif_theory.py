import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plt_scc_qif_theory(mu, tau_a, delta, D, ratio):
    home = os.path.expanduser('~')
    corr_theory = []
    corr_sim = []
    cvs = []

    k_min = 1
    k_max = 5
    k_range_theory = range(k_min, k_max+1) #np.linspace(k_min, k_max, 30, endpoint=True)
    k_range_sim = range(k_min, k_max+1)
    tau_ns = [0.1, 1., 10]

    # Calculate serial correlation coefficient for colored noise
    for tau_n in tau_ns:
        print(tau_n)
        corr_theory_sub = []
        corr_sim_sub = []
        data_file = home + "/Data/QIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
            mu, tau_a, tau_n, D, delta, ratio)
        data = np.loadtxt(data_file, max_rows=24_000)
        t, a, eta, chi, chi2 = np.transpose(data)
        t_mean = np.mean(t)
        delta_t = [x - t_mean for x in t]
        t_det = np.pi/np.sqrt(mu)

        for k in k_range_theory:
            corr_theory_sub.append(fc.cnoise_QIF_scc(t_det, tau_n, mu, k))
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
        ax.scatter(k_range_sim, corr_sim[i], c = color, edgecolor = "k", s=20, zorder=2)

    ax.set_ylabel(r"$\rho_k$")
    ax.set_xlabel("k")
    ax.set_xlim([k_min-0.2, k_max+0.2])

    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # Use this to emphasize the discrete color values
    cbar = f.colorbar(s_map, cax=cax, format='%.0e')
    cbar.set_ticks(colorparams)
    cbar.ax.set_ylabel(r"$\tau_\eta \times 1$")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_yticklabels([r'$10^{{{:.0f}}}$'.format(np.log(tau)/np.log(10)) for tau in tau_ns])

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
    plt.savefig(home + "/Data/QIF/white/plots/scc_theory_sim_mu{:.1f}_taua{:.1f}_delta{:.1f}_D{:.1f}_ratio{:.2f}.pdf".format(mu, tau_a, delta, D, ratio), transparent=True)
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        mu = 5
        tau_a = 2
        gamma = 1
        D=0.1
        ratio = 0.0
        tau_a = 0
        delta = 0
        plt_scc_qif_theory(mu, tau_a, delta, D, ratio)

