import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc


def plot_cv_r0():
    home = os.path.expanduser('~')

    mus = []
    cvs = []
    cvs_theory = []
    for k in range(1,30):
        tau_n = 0.1
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
        cvs.append(np.mean([t*t for t in delta_t]))
        if(tau_n == 1): tau_n = 1.01
        delta_t_theory1 = sigma*np.exp(-t_det)*(np.exp((1.-1./tau_n)*t_det)-1)/(np.power(mu - 1, 2)*(1-1/tau_n))

        delta_t_theory2 = (1/np.power(mu-1, 2))*2*sigma*np.exp(-2*t_det)*((np.exp(2*t_det) -1)/(2*(1+1/tau_n)) - (np.exp((1-1/tau_n)*t_det) -1)/((1+1/tau_n)*(1-1/tau_n)))
        cvs_theory.append(delta_t_theory2/(t_det**2))

    f, ax = plt.subplots(1, 1, figsize=(5, 4.8*5/6.4))

    ax.scatter(mus, cvs, c='k')
    ax.plot(mus, cvs_theory)
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in', which = "minor")
    ax.set_ylabel(r"$C_V^2$")
    ax.set_xlabel("$\mu$")
    plt.tight_layout()
    #plt.savefig(home + "/Data/cLIF_plots/cv_theorie_taua{:.1f}_delta{:.1f}_D{:.1f}.pdf".format(tau_a, tau_n, delta, D), transparent=True)
    plt.show()


if __name__ == "__main__":
    plot_cv_r0()
