import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc

def plt_mean_t_qif(tau_n, D):
    tau_a = 0
    delta = 0
    ts_dif = []
    ts_dif_theo = []
    taus = []
    mus = []
    error = []
    for k in range(25):
        #tau_n = 0.2*(k+1)
        mu = 0.2*(k+1)
        sigma = D / tau_n
        t_det = np.pi / np.sqrt(mu)
        w = np.pi / t_det
        home = os.path.expanduser('~')
        data_file = home + "/Data/QIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
        mu, tau_a, tau_n, D, delta, 0)

        data = np.loadtxt(data_file)
        t, a, eta, chi, chi2 = np.transpose(data)
        t_dif = (-t_det + t)/t_det
        error.append(fc.estimate_error(t_dif, np.mean, 20))
        ts_dif.append(np.mean(t_dif))
        taus.append(tau_n)
        mus.append(mu)
        c_1 =  1./(2*mu)
        delta_w  = -(sigma/2.)*(c_1**2)*(w*tau_n**2)/(1. + (tau_n*w)**2)
        ts_dif_theo.append(-2*(delta_w/(w+delta_w)))
        print("$\mu$", mu, r"$\tau$", tau_n, "$C_v$", np.sqrt(sigma)/t_det)

    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    ax.set_ylabel(r"$\Delta T/T$")
    if min(ts_dif) < min(ts_dif_theo):
        min_y = min(ts_dif)*0.8
    else:
        min_y = min(ts_dif_theo)*0.8

    if max(ts_dif) > max(ts_dif_theo):
        max_y = max(ts_dif) * 1.2
    else:
        max_y = max(ts_dif_theo) * 1.2

    #ax.set_ylim([min_y, max_y])
    ax.set_ylim(0.00001, 1)
    ax.set_xlim([0, 5])
    ax.set_yscale("log")
    #ax.set_xlabel(r"$\tau_\eta$")
    #ax.scatter(taus, ts_dif)
    #ax.plot(taus, ts_dif_theo, c="k")
    ax.set_xlabel(r"$\mu$")
    ax.errorbar(mus, ts_dif, yerr=error, fmt='o', label = "$\sigma^2 = {:.2f}$".format(D/tau_n))
    ax.plot(mus, ts_dif_theo, c="k")
    ax.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/QIF/white/plots/spike_time_derivation_taun{:.1f}_D{:.1f}.pdf".format(tau_n, D))
    plt.show()
    return 1

if __name__ == "__main__":
    #mus = [0.25, 1, 1.75, 5]
    #D = 0.1
    #for mu in mus:
    #    plt_mean_t_qif(mu, D)

    plt_mean_t_qif(0.1, 0.1)