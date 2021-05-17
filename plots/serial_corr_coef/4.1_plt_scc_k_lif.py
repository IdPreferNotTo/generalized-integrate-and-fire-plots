import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from utilites import plot_parameters as utl
from scipy import integrate
import cmath

def LIF_gnoise_scc(t_det, mu, tau_a, delta, tau_n, Dn, Dw, k):
    if tau_a == 0:
        alpha = 0
        a_det = 0
        nu = 1
        def pa(i): return 0
    else:
        alpha = np.exp(-t_det / tau_a)
        beta = np.exp(-t_det / tau_n) if tau_n != 0 else 0
        a_det = delta / (tau_a * (1 - alpha))
        def h(t):
            return fc.lif_varprc(t, t_det, a_det, mu, tau_a, delta) * np.exp(-t / tau_a)

        zi = integrate.quad(h, 0, t_det)[0]
        nu = 1. - (a_det / tau_a) * zi

        def a(i):
            return alpha * (1 - alpha * alpha * nu) * (1 - nu) * np.real(cmath.exp((i - 1) * cmath.log(alpha * nu)))
        c = 1 + alpha ** 2 - 2 * alpha ** 2 * nu
        def pa(i): return -a(i) / c

    if tau_n == 0:
        beta = 0
        chi1 = 0
        chi0 = 1
    else:
        beta = np.exp(-t_det / tau_n)
        chi1 = 0
        chi0 = 0
        ts = np.linspace(0, t_det, 100)
        dt = t_det / 100
        for t1 in ts:
            for t2 in ts:
                Zt1 = fc.lif_varprc(t1, t_det, a_det, mu, tau_a, delta)
                Zt2 = fc.lif_varprc(t2, t_det, a_det, mu, tau_a, delta)
                chi1 += Zt1 * Zt2 * (Dn/tau_n - 2*np.sqrt(Dn*Dw)) * np.exp(-(t_det + t2 - t1) / tau_n) * dt * dt
                chi0 += Zt1 * Zt2 * (Dn/tau_n - 2*np.sqrt(Dn*Dw)) * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            Zt1 = fc.lif_varprc(t1, t_det, a_det, mu, tau_a, delta)
            chi0 += (2 * Dw) * Zt1 ** 2 * dt

    def pn(i): return beta ** (i-1) * chi1 / chi0

    A = 1 + (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta)/((alpha * nu - beta)) * pn(1) - alpha * nu * beta
    B = (1. - (alpha * nu) ** 2) * (1. - alpha * beta)*((alpha - beta)) / ((1 + alpha ** 2 - 2 * alpha ** 2 * nu)*(alpha * nu - beta))
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta
    return (A/C) * pa(k) + (B/C) * pn(k)


def plt_scc_lif(taua, taun, mu, delta, Dn, Dw):
    home = os.path.expanduser('~')

    # SSC FROM SIMULATIONS
    data_file = home + "/Data/LIF/data/mu{:.2f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}_0.txt".format(mu, taua, delta, taun, Dn, Dw)
    data = np.loadtxt(data_file)
    isis, deltaa, eta, lin_resp = np.transpose(data)
    t_mean = np.mean(isis)
    CV = np.std(isis)/t_mean
    print(CV)
    k_correlations = []
    ks = np.arange(1, 10)
    std = fc.k_corr(isis, isis, 0)
    for k in ks:
        rhok = fc.k_corr(isis, isis, k)
        k_correlations.append(rhok/std)

    # SCC FROM THEORY
    k_correlations_theory = []
    for k in ks:
        k_corr = LIF_gnoise_scc(t_mean, mu, taua, delta, taun, Dn, Dw, k)
        k_correlations_theory.append(k_corr)

    f, ax = plt.subplots(1, 1, figsize=utl.adjust_plotsize(0.9, 0.5))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in', labelsize=utl.labelsize)
    ax.scatter(ks, k_correlations, label = "Simulation $C_V = {:.2f}$".format(CV))
    ax.plot(ks, k_correlations_theory, label = "Theory")
    ax.set_ylabel(r"$\rho_k$", rotation=0, fontsize=utl.fontsize)
    ax.set_xlabel("$k$", fontsize=utl.fontsize)
    ax.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/LIF/plots/scc_full_mu{:.1f}_taua{:.1f}_taun{:.1f}_delta{:.1f}_Dw{:.2f}_Dn{:.2f}.pdf".format(mu, tau_a, tau_n, delta, Dw, Dn))
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        for j in range(1):
            gamma = 1
            mu = 5
            tau_a = 2
            tau_n = 1.0 #0
            delta_a = 2
            Dn = 0.33/4#0
            Dw = 0.50/4
            plt_scc_lif(tau_a, tau_n, mu, delta_a, Dn, Dw)

