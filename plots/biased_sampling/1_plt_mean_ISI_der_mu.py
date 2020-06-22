import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
import scipy.integrate as integrate
from scipy.special import erfc
import matplotlib.ticker as ticker
import os

def plt_mean_ISI_der_mu():
    home = os.path.expanduser('~')

    tau_n = 1.0
    v_r = 0
    vt = 1

    mus = []
    plot_data = []
    plot_data2 = []
    plot_data3 = []
    plot_data4 = []
    plot_data5 = []
    plot_data6 = []
    plot_error_data = []

    cv = False
    cv_line = 0

    for k in range(1,30):
        tau_n = 5.0
        mu = 1.1 + 0.1*k
        sigma = 0.1
        D = sigma*tau_n
        mus.append(mu)

        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_0.txt".format(mu, tau_n, D)

        print(data_file)
        t_det = fc.read_t_det(data_file)
        data = np.loadtxt(data_file)
        t, a, chi, eta, chi2 = np.transpose(data)
        delta_t = [(x - t_det)/t_det for x in t]
        plot_data.append(sum(delta_t)/len(delta_t))
        # Eq. 265
        plot_data2.append((-np.power(mu-1., -1)*np.exp(-t_det)*np.mean(chi2))/t_det)
        # Eq. 268
        plot_data3.append((-np.power(mu-1., -1)*np.exp(-t_det)*np.mean(chi) - np.power(mu-1., -1)*fc.k_corr([t*t_det for t in delta_t], eta, 0))/t_det)
        # Eq. 274
        plot_data4.append((-np.power(mu-1., -1)*np.exp(-t_det)*np.mean(chi) + np.power(mu-1., -2)*np.exp(-t_det)*fc.k_corr(chi, eta, 0) - np.power(mu-1., -3)*np.exp(-t_det)*fc.k_corr(chi, [x**2 for x in eta], 0) + np.power(mu-1., -4)*np.exp(-t_det)*fc.k_corr(chi, [x**3 for x in eta], 0))/t_det)
        if(tau_n == 1): tau_n = 1.001
        #boundaries at infinity
        #plot_data5.append((-np.power(mu-1., -2)*sigma*np.exp(-t_det)*(np.exp((1-1/tau_n)*t_det)-1)*np.power(1-1/tau_n, -1) + np.power(mu-1., -2)*np.exp(-t_det)*fc.k_corr(chi, eta, 0))/t_det)
        #plot_data6.append((-np.power(mu-1., -2)*sigma*np.exp(-t_det)*(np.exp((1-1/tau_n)*t_det)-1)*np.power(1-1/tau_n, -1) + np.power(mu-1., -2)*np.exp(-t_det)*(sigma / (1. + 1. / tau_n)) * np.exp(-t_det / tau_n) * (np.exp((1 + 1 / tau_n) * t_det) - 1))/t_det)
        #boundaries at -f(v_t) and infinity
        x = (mu-1)/np.sqrt(2*sigma)
        c = ((mu-1.)/2)*erfc(-x) + np.sqrt(sigma/(2*np.pi))*np.exp(-x*x)
        term1 = -np.power(2*c*(mu - 1.), -1) * erfc(-x) * sigma * np.exp(-t_det) * (np.exp((1 - 1 / tau_n) * t_det) - 1) * np.power(1 - 1 / tau_n, -1)
        term2 = np.power(2*c*(mu - 1.), -1) * erfc(-x) * np.exp(-t_det)*(sigma / (1. + 1. / tau_n)) * np.exp(-t_det / tau_n) * (np.exp((1 + 1 / tau_n) * t_det) - 1)
        plot_data5.append((term1 + np.power(mu - 1., -2) * np.exp(-t_det) * fc.k_corr(chi, eta, 0)) / t_det)
        plot_data6.append((term1 + term2) / t_det)

        plot_error_data.append(np.sqrt(np.std(delta_t)/len(delta_t)))
        if(np.sqrt(np.var(t))/np.mean(t) < 0.4 and cv == False):
            cv = True
            cv_line = mu


    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    #ax.errorbar(mus_inset, plot_data_inset, yerr=plot_error_data_inset, c="C0")
    ax.errorbar(mus, plot_data, c="k", yerr= plot_error_data, fmt = "o")
    ax.set_ylim([0, 0.1])
    ax.plot(mus[5:], plot_data2[5:], label="Eq. 265")
    ax.plot(mus, plot_data3, label="Eq. 268")
    ax.plot(mus, plot_data4, label ="Eq. 274")
    ax.plot(mus, plot_data5, label =r"subst. $\langle \eta(t_i) \rangle$")
    ax.plot(mus, plot_data6, label =r"subst. $\langle \eta(t_i)\eta(t_j) \rangle$")
    #ax.plot(mus[1:], data_chi1[1:], c="C1", label = r"Theory $\int_{-f(v_t)}^\infty \mathrm{d}\eta_0$")
    #ax.plot(mus[1:], data_chi2[1:], c="C2", label = r"Theory $\int_{-\infty}^\infty \mathrm{d}\eta_0$")
    ax.set_xlabel("$\mu$")
    ax.set_ylabel(r"$\langle \delta T \rangle / T$")
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    #ax.set_yscale("log")
    #ax.set_ylim([0.00, 0.12])
    ax.set_xlim([1,4])
    #ax.set_yticks([-0.05, 0., 0.05, 0.1])
    ax.set_xticks([1, 2, 3, 4])
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    ax.axvline(cv_line, c = "C7", ls = "--")
    ax.axvspan(1, cv_line, facecolor="C7", alpha=0.5)
    #plt.legend()

    plt.tight_layout()
    #plt.savefig(home + "/Data/LIF/red/plots/delta_t_mu_D{:.1f}_tau_n{:.2f}_better.pdf".format(D, tau_n))
    plt.show()

    return 1

if __name__ == "__main__":
    plt_mean_ISI_der_mu()