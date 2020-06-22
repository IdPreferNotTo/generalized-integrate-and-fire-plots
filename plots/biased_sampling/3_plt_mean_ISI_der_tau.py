import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
import scipy.integrate as integrate
from scipy.special import erfc
import matplotlib.ticker as ticker
import os

def plt_mean_ISI_der_sigma():
    home = os.path.expanduser('~')

    v_r = 0
    vt = 1

    taus = []
    plot_data = []
    plot_data2 = []
    data_chi1 = []
    data_chi2 = []
    plot_error_data = []

    for k in range(1, 20):
        mu = 2.0
        tau_n = 0.1*k
        if tau_n == 1:
            continue
        sigma = 0.1
        D = sigma*tau_n
        taus.append(tau_n)

        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_0.txt".format(mu, tau_n, D)

        print(data_file)
        t_det = fc.read_t_det(data_file)
        data = np.loadtxt(data_file)
        t, a, chi, eta, chi2 = np.transpose(data)
        delta_t = [(x - t_det)/t_det for x in t]
        plot_data.append(sum(delta_t)/len(delta_t))
        plot_error_data.append(np.sqrt(np.var(delta_t)/len(delta_t)))
        plot_data2.append(-(1/(mu-1.))*np.exp(-t_det)*sum(chi2)/len(chi2))


        f1 = lambda t: np.exp(t - t_det) * np.exp(-t / tau_n)
        F1 = integrate.quad(f1, 0, t_det)
        f2 = lambda t: np.exp(t - t_det) * np.exp((t - t_det) / tau_n)
        F2 = integrate.quad(f2, 0, t_det)

        x = (mu - 1.) / np.sqrt(2 * sigma)
        corr_full_term = 1 / ((mu - 1.) + (2. / erfc(-x)) * np.sqrt(sigma / (2 * np.pi)) * np.exp(-x * x))

        corr0 = (sigma / np.power(mu - 1., 2)) * F1[0]

        corr1 = (sigma / np.power(mu - 1., 2)) * F2[0]

        data_chi1.append(corr_full_term * (mu - 1) * (-corr0 + corr1) / t_det)
        data_chi2.append((-corr0 + corr1) / t_det)

    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    ax.errorbar(taus, plot_data, yerr= plot_error_data, fmt = "o", c="C0", label = "Simulation")
    #ax.scatter(taus, plot_data2)
    ax.plot(taus, data_chi1, c="C1", label = r"Theory $\int_{-f(v_t)}^\infty \mathrm{d}\eta_0$")
    ax.plot(taus, data_chi2, c="C2", label = r"Theory $\int_{-\infty}^\infty \mathrm{d}\eta_0$")
    ax.set_xlabel(r"$\tau_\eta$")
    ax.set_ylabel(r"$\langle \delta T \rangle/T$")
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    plt.legend()

    plt.tight_layout()
    #plt.savefig(home + "/Data/LIF/red/plots/delta_t_tau_mu{:.2f}_sigma{:.2f}.pdf".format(mu, D/tau_n))
    plt.show()

    return 1

if __name__ == "__main__":
    plt_mean_ISI_der_sigma()