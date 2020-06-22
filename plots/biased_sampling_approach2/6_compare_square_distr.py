import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import scipy.stats as stats
import os


def plt_compare_distr(mu, tau_n, D):
    home = os.path.expanduser('~')
    f, axis = plt.subplots(2, 2)
    for ax_row in axis:
        for ax in ax_row:
            ax.tick_params(direction='in')
    for i in range(4):
        tau_n = [0, 0.1, 1, 2.9][i]
        if (tau_n == 0.0): tau_n = 0.001
        sigma = D / tau_n
        gamma=1
        vR = 0
        vT = 1
        f_vT = mu-gamma*vT
        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_{:d}.txt".format(mu,
                                                                                                                 tau_n,
                                                                                                                 D, 0)
        print(data_file)
        t_det = np.log((mu - vR) / (mu - vT))
        data = np.loadtxt(data_file, max_rows=50_000)
        t, a, eta, chi, chi2 = np.transpose(data)
        if(tau_n == 0.001): chi = [1_000*np.sqrt(2*D)*x for x in chi]

        chi = [-x*np.exp(-t_det)/((mu - 1.)*(t_det)) for x in chi]
        chi2 = [-x*np.exp(-2*t_det)/((mu-1.)*t_det) for x in chi2]
        delta_t = [(x - t_det) / t_det for x in t]
        delta_t = [t**2 for t in delta_t]

        chi = [x+y for (x,y) in zip(chi, chi2)]
        chi = [x ** 2 for x in chi]
        mean_delta_t = np.mean(delta_t)
        mean_chi = np.mean(chi)
        print(mean_delta_t)
        print(mean_chi)

        if(int(i/2) == 0 and i%2 == 0):
            density = stats.gaussian_kde(delta_t)
            n, x, _ = axis[int(i / 2)][i % 2].hist(delta_t, bins=np.linspace(-0.1, 1, 50), density=True, alpha=0.5, label = "{:.2e}".format(mean_delta_t))
            #axis[int(i / 2)][i % 2].plot(x, density(x), color=["k", "C0", "C1", "C2"][i], label = "{:.2e}".format(mean_delta_t))

            density = stats.gaussian_kde(chi)
            n, y, _ = axis[int(i / 2)][i % 2].hist(chi, bins=np.linspace(-0.1, 1, 50), density=True, alpha=0.5, label = "{:.2e}".format(mean_chi))
            #axis[int(i / 2)][i % 2].plot(y, density(y), color=["k", "C0", "C1", "C2"][i], ls="--", label = "{:.2e}".format(mean_chi))
            axis[int(i / 2)][i % 2].legend()
        else:
            density = stats.gaussian_kde(delta_t)
            n, x, _ = axis[int(i/2)][i%2].hist(delta_t, bins=np.linspace(-0.1, 1, 50), density=True, alpha=0.5, label = "{:.2e}".format(mean_delta_t))
            #axis[int(i/2)][i%2].plot(x, density(x), color=["k", "C0", "C1", "C2"][i], label = "{:.2e}".format(mean_delta_t))

            density = stats.gaussian_kde(chi)
            n, y, _ = axis[int(i/2)][i%2].hist(chi, bins=np.linspace(-0.1, 1, 50), density=True, alpha=0.5, label = "{:.2e}".format(mean_chi))
            #axis[int(i/2)][i%2].plot(y, density(y), color=["k", "C0", "C1", "C2"][i], ls = "--", label = "{:.2e}".format(mean_chi))
            axis[int(i / 2)][i % 2].legend()

    axis[0][0].set_ylabel("$p(\delta T / T^2)$")
    axis[1][0].set_ylabel("$p(\delta T / T^2)$")
    axis[1][0].set_xlabel("$(\delta T / T)^2$")
    axis[1][1].set_xlabel("$(\delta T / T)^2$")
    plt.savefig(home + "/Data/LIF/red/plots/delta_t_square_distributions_mu{:.2f}_D{:.2f}.pdf".format(mu, D))
    plt.show()
    return 1


if __name__ == "__main__":
    mu = 2.
    tau_n = 0.1
    D = 0.1
    plt_compare_distr(mu, tau_n, D)