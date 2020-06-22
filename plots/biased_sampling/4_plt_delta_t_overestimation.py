import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
import os

def plt_delta_t_overestimation():
    home = os.path.expanduser('~')
    data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_0.txt".format(mu, tau_n, D)

    print(data_file)
    t_det = fc.read_t_det(data_file)
    print(t_det)
    data = np.loadtxt(data_file)
    t, a, chi, eta, chi2 = np.transpose(data[0:5_000])
    delta_t = [(x - t_det)/t_det for x in t]
    print(max(delta_t))
    chi2 = [-(np.exp(-t_det)/(mu-1.))*x/t_det for x in chi2]
    chi3 = [-(np.exp(-t_det)/(mu-1.))*x/t_det for x in chi]

    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.scatter(delta_t, chi2, s=5, label = "$x = T +\delta T_i$")
    ax.scatter(delta_t, chi3, s=5, label = "$x = T$")
    ax.set_xlim([min(chi2), max(chi2)])
    ax.set_ylim([min(chi2), max(chi2)])
    ax.plot([min(chi2), max(chi2)], [min(chi2), max(chi2)], ls="--", c=".3")
    ax.set_xlabel(r"$\langle \delta T\rangle/T$")
    ax.set_ylabel(r"$-\int_0^{x} Z(t)\eta(t_i + t)/T$")
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.legend()
    plt.tight_layout()
    #plt.savefig(home + "/Data/LIF/red/plots/delta_t_overstimation_mu{:.2f}_tau_n{:.2f}_D{:.3f}.pdf".format(mu, tau_n, D))
    plt.show()
    return 1

if __name__ == "__main__":
    mu = 1.5
    tau_n = 1.0
    D = 0.1000
    plt_delta_t_overestimation()