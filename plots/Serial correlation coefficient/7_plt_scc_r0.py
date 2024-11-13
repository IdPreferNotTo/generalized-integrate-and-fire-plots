import numpy as np
import matplotlib.pyplot as plt
import os


def plt_scc_firing_rate(k, tau_a, tau_n, gamma, delta, v_t):
    home = os.path.expanduser('~')

    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    scc_file_path = home + "/Data/scc_firing_rate_data/scc{:d}_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.txt".format(k, tau_a, tau_n, delta, D)
    scc_data = np.loadtxt(scc_file_path)
    ts, scc_adap_cnoise_sim, scc_adap_theory, scc_adap_cnoise_theory =  np.transpose(scc_data) # scc = serial corr. coef., w = white, c = colored, t= theory, s = simulation
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    ax.plot([1/t for t in ts], scc_adap_cnoise_theory, label = "colored", c="k")
    ax.plot([1/t for t in ts], scc_adap_theory, label = "white" , c="k", ls = "--")
    ax.scatter([1/t for t in ts], scc_adap_cnoise_sim, s=5, label = "simulation")
    #ax.legend()
    ax.set_xscale("log")
    ax.set_xlim([0.01 , 10])
    ax.set_ylabel(r"$\rho_{:d}$".format(k))
    ax.set_xlabel("firing rate")
    plt.tight_layout()
    plt.savefig(home + "/Data/cLIF_plots/scc{:d}_theorie_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.pdf".format(k, tau_a, tau_n, delta, D),
                transparent=True)
    plt.show()

if __name__ == "__main__":
    for i in [1, 2]:
        for j in [1, 10]:
            for m in [1, 10]:
                k = i
                tau_a = 10
                tau_n = j
                gamma = 1
                delta = m
                D = 0.10
                v_t = 1
                plt_scc_firing_rate(k, tau_a, tau_n, gamma, delta, v_t)

