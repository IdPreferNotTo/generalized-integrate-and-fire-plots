import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plt_scc_gif_theory(gamma, mu, tau_a, delta, beta, tau_w, D, ratio):
    home = os.path.expanduser('~')

    corr_adap_cnoise_theory = []
    corr_adap_cnoise_sim = []
    corr_adap_theory = []
    corr_adap_sim = []
    cvs = []

    beta0 = np.sqrt(2*D)/(1. + ratio)
    beta1 = ratio*beta0
    k_min = 1
    k_max = 5
    k_range_theory = range(k_min, k_max +1) #np.linspace(k_min, k_max, 30, endpoint=True)
    k_range_sim = range(k_min, k_max+1)
    tau_ns = [0.08, 0.32, 1.28, 5.12]

    # Calculate serial correlation coefficient for white noise
    data_file = home + "/Data/GIF/green/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
        mu, tau_a, 0.01, D, delta, ratio)
    data = np.loadtxt(data_file)
    t, a, eta, chi, chi2 = np.transpose(data)
    t_mean = np.mean(t)
    delta_t = [x - t_mean for x in t]
    w_det = fc.read_w_det(data_file)
    t_det = fc.read_t_det(data_file)

    for k in k_range_theory:
        corr_adap_theory.append(fc.adap_gnoise_GIF_scc(t_det, w_det, gamma, mu, tau_a, delta, beta, tau_w, 0, beta0, beta1, k))
    for k in k_range_sim:
        c0 = fc.k_corr(delta_t, delta_t, 0)
        ck = fc.k_corr(delta_t, delta_t, k)
        corr_adap_sim.append(ck / c0)

    # Calculate serial correlation coefficient for colored noise
    for tau_n in tau_ns:
        print(tau_n)
        corr_adap_cnoise_theory_sub = []
        corr_adap_cnoise_sim_sub=[]
        data_file = home + "/Data/GIF/green/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
            mu, tau_a, tau_n, D, delta, ratio)
        data = np.loadtxt(data_file)
        t, a, eta, chi, chi2 = np.transpose(data)
        t_mean = np.mean(t)

        delta_t = [x - t_mean for x in t]
        w_det = fc.read_w_det(data_file)
        t_det = fc.read_t_det(data_file)

        for k in k_range_theory:
            print("k: ", k)
            corr_adap_cnoise_theory_sub.append(fc.adap_gnoise_GIF_scc(t_det, w_det, gamma, mu, tau_a, delta, beta, tau_w, tau_n, beta0, beta1, k))
        for k in k_range_sim:
            c0 =fc.k_corr(delta_t, delta_t, 0)
            ck =fc.k_corr(delta_t, delta_t, k)
            corr_adap_cnoise_sim_sub.append(ck/c0)
        cvs.append(np.sqrt(c0) / t_mean)
        corr_adap_cnoise_theory.append(corr_adap_cnoise_theory_sub)
        corr_adap_cnoise_sim.append(corr_adap_cnoise_sim_sub)

    f, (ax, cax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[15,1]}, figsize=(5, 9 / 3))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')

    # Which parameter should we represent with color?
    colorparams = tau_ns
    colormap = cm.viridis
    normalize = mcolors.SymLogNorm(0.1, vmin=np.min(colorparams), vmax=np.max(colorparams))

    for i, tau_n in enumerate(tau_ns):
        color = colormap(normalize(tau_n))
        ax.plot(k_range_theory, corr_adap_cnoise_theory[i], label ="colored", c = color, lw =1, zorder=1)
        ax.scatter(k_range_sim, corr_adap_cnoise_sim[i], c = color, s=20, edgecolors="k", zorder=2)

    ax.plot(k_range_theory, corr_adap_theory, label ="white", c = "k", ls = "--", zorder=1)
    ax.scatter(k_range_sim, corr_adap_sim, c='k', s=20, zorder=2)
    ax.set_ylabel(r"$\rho_k$")
    ax.set_xlabel("k")
    ax.set_xlim([k_min, k_max])

    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # Use this to emphasize the discrete color values
    cbar = f.colorbar(s_map, cax=cax, format='%.2f')
    cbar.set_ticks(colorparams)
    cbar.ax.set_ylabel(r"$\tau_\eta$")
    cbar.ax.yaxis.set_label_position("left")

    cax_cv = cax.twinx()
    cax_cv.set_ylim(0, 3)
    cax_cv.set_yticks([0, 1, 2, 3])
    exp = 0
    for i in range(3):
        if (min(cvs)) < 10**(-i):
            exp = i+1

    cax_cv.set_yticklabels(["{:.0f}".format((10**exp)*cv) for cv in cvs])
    cax_cv.set_ylabel(r"$C_V \times 10^{{-{:d}}}$".format(exp))

    #ax.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/GIF/green/plots/scc_theory_sim_mu{:.1f}_taua{:.1f}_delta{:.1f}_D{:.1f}_ratio{:.2f}.pdf".format(mu, tau_a, delta, D, ratio), transparent=True)
    plt.show()

if __name__ == "__main__":
    for i in range(5):
        mu = [10, 11.75, 20, 2.12, 1.5][i]
        tau_a = [10, 10, 10, 1, 1][i]
        gamma = 1
        delta = [1, 1, 1, 10, 9][i]
        beta = [3, 3, 1.5, 1.5, 1.5][i]
        tau_w = 1.5
        D= [10**(-4), 10**(-4), 10**(-4), 10**(-4), 10**(-5)][i]
        ratio = -0.68
        plt_scc_gif_theory(gamma, mu, tau_a, delta, beta, tau_w, D, ratio)


