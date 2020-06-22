import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
import scipy.integrate as integrate
from scipy.special import erfc
import matplotlib.ticker as ticker
import os

def plt_mean_ISI_der_sigma():
    home = os.path.expanduser('~')

    tau_n = 0.1
    v_r = 0
    vt = 1

    sigmas = []
    plot_data = []
    data_chi1 = []
    data_chi2 = []
    plot_error_data = []

    cv = False
    cv_line = 0

    for k in range(1, 20):
        mu = 2.0
        sigma = 0.05*k
        D = sigma*tau_n
        sigmas.append(sigma)

        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_1.txt".format(mu, tau_n, D)

        print(data_file)
        t_det = fc.read_t_det(data_file)
        data = np.loadtxt(data_file)
        t, a, chi, eta, chi2 = np.transpose(data)
        delta_t = [(x - t_det)/t_det for x in t]
        plot_data.append(sum(delta_t)/len(delta_t))
        plot_error_data.append(np.std(delta_t)/np.sqrt(len(delta_t)))

        f1 = lambda t: np.exp(t-t_det)*np.exp(-t/tau_n)
        F1 = integrate.quad(f1, 0, t_det)
        f2 = lambda t: np.exp(t-t_det)*np.exp((t-t_det)/tau_n)
        F2 = integrate.quad(f2, 0, t_det)


        x = (mu - 1.) / np.sqrt(2 * sigma)
        corr_full_term = 1/((mu-1.) + (2./erfc(-x))*np.sqrt(sigma/(2*np.pi))*np.exp(-x*x))

        corr0 = (sigma/np.power(mu-1., 2))*F1[0]

        corr1 = (sigma/np.power(mu-1., 2))*F2[0]

        data_chi1.append(corr_full_term*(mu-1)*(-corr0 + corr1)/t_det)
        data_chi2.append((-corr0 + corr1)/t_det)
        if(np.sqrt(np.var(t))/np.mean(t) > 0.4 and cv == False):
            cv = True
            cv_line = sigma

    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    ax.axvline(np.power((mu-1), 2), ls='--', c='C7')
    ax.annotate("$\sigma^2 = f(v_t)^2$", xy=(np.power((mu-1), 2)+0.01, 0.003), rotation=90)
    ax.errorbar(sigmas, plot_data, yerr= plot_error_data, fmt = "o", c="C0", label = "Simulation")
    #ax.plot(sigmas, data_chi1, c="C1", label = r"Theory $\int_{-f(v_t)}^\infty \mathrm{d}\eta_0$")
    #ax.plot(sigmas, data_chi2, c="C2", label = r"Theory $\int_{-\infty}^\infty \mathrm{d}\eta_0$")
    ax.set_xlabel("$\sigma^2$")
    ax.set_ylabel(r"$\langle \delta T \rangle/T$")
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.axvline(cv_line, c="C7", ls="--")
    ax.axvspan(cv_line, max(ax.get_xlim()), facecolor="C7", alpha=0.5)
    ax.tick_params(direction='in')
    plt.legend()

    plt.tight_layout()
    #plt.savefig(home + "/Data/LIF/red/plots/delta_t_sigma_mu{:.2f}_tau_n{:.2f}.pdf".format(mu, tau_n))
    plt.show()

    return 1

if __name__ == "__main__":
    plt_mean_ISI_der_sigma()