import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import gridspec

from utilites import functions as fc
from utilites import plot_parameters as utl

def k_corr(data1, data2, k):
    # Get two arbitrary data set and calculate their correlation with lag k.
    mean_d1 = np.mean(data1)
    mean_d2 = np.mean(data2)
    data1 = [x - mean_d1 for x in data1]
    data2 = [x - mean_d2 for x in data2]

    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])

if __name__ == "__main__":
    home = os.path.expanduser("~")
    file = home + "/Desktop/spike.npy"
    data = np.load(file, allow_pickle=True)

    isis_per_neuron = []
    for spike_times in data:
        t_tmp = 0
        isis = []
        for t in spike_times:
            isi = t - t_tmp
            isis.append(isi)
            t_tmp = t
        isis_per_neuron.append(list(isis))

    var = 0
    mean = 0
    for isis in isis_per_neuron:
        mean += np.mean(isis)
        var += np.var(isis)
    var = var/5_000
    mean = mean/5_000
    print(np.sqrt(var)/mean)

    ks = np.arange(1, 6)
    k_correlations = [[], [], [], [], []]
    for isis in isis_per_neuron:
        var = k_corr(isis, isis, 0)
        for k in ks:
            covar = k_corr(isis, isis, k)
            k_correlations[k-1].append(covar/var)

    fig = plt.figure(tight_layout=True, figsize=utl.adjust_plotsize(0.8))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    k_mean_correlations = []
    for correlations in k_correlations:
        k_mean_correlations.append(np.mean(correlations))
    ax.scatter(ks, k_mean_correlations, color="C3", s=25, edgecolors="k", lw=1, zorder=3, label="Network average")
    ax.plot(ks, k_mean_correlations, color="k", lw=1)

    for k, corrs in zip(ks, k_correlations):
        for corr in corrs[:100]:
            ax.scatter(k, corr, color="C7", s=10, lw=1, zorder=2)
    ax.scatter(5, corr, color="C7", s=10, lw=1, zorder=2, label="Single neuron")
    ax.axhline(0, ls="--", c="C7")

    ax.legend(prop={"size": 7}, loc=4, ncol=1, framealpha=1., edgecolor="k")
    leg = ax.get_legend()
    leg.get_frame().set_linewidth(0.5)
    ax.set_xlabel("$k$")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylabel(r"$\rho_k$")

    plt.savefig(home + "/Data/SCC/Plots_paper/scc_brunel_network.pdf")
    plt.show()
