import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import gridspec

from utilites import functions as fc
from utilites import plot_parameters as utl
from scipy import integrate
import cmath

def LIF_gnoise_scc(t_det, mu, tau_a, delta, tau_n, bn, bw, k):
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
                chi1 += Zt1 * Zt2 * (np.power(bn, 2)/(2*tau_n) + bn*bw/tau_n) * np.exp(-(t_det + t2 - t1) / tau_n) * dt * dt
                chi0 += Zt1 * Zt2 * (np.power(bn, 2)/(2*tau_n) + bn*bw/tau_n) * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            Zt1 = fc.lif_varprc(t1, t_det, a_det, mu, tau_a, delta)
            chi0 += np.power(bw, 2) * Zt1 ** 2 * dt

    def pn(i): return beta ** (i-1) * chi1 / chi0

    A = 1 + (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta)/((alpha * nu - beta)) * pn(1) - alpha * nu * beta
    B = (1. - (alpha * nu) ** 2) * (1. - alpha * beta)*((alpha - beta)) / ((1 + alpha ** 2 - 2 * alpha ** 2 * nu)*(alpha * nu - beta))
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta
    return (A/C) * pa(k) + (B/C) * pn(k)


def fouriertransformation(w, ts):
    f = 0.
    x = 1.
    for t in ts:
        f += x*np.exp(1j * w * t)
        x *= np.exp(1j * w * t)
    return f


def slice(data,n):
    for i in range(0, len(data), n):
        yield data[i:i+n]


def power_spectrum(spiketrain):
    Sfs = []
    ws = np.logspace(-2, 2, 100)

    for w in ws:
        xfs = []
        for xs in slice(spiketrain, 100):
            T = sum(xs)
            xf = fouriertransformation(w, xs)
            xfs.append(xf/np.sqrt(T))
        var = np.var(xfs)
        Sfs.append(var)
    return ws, Sfs


def plt_scc_lif(taua, mu, delta, bn, bv):
    home = os.path.expanduser('~')

    # SSC FROM SIMULATIONS
    fig = plt.figure(tight_layout=True, figsize=utl.adjust_plotsize(1., 0.5))
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0:1])
    ax2 = fig.add_subplot(gs[1:2])

    tauns = [0.01, 0.1, 1.0]
    for taun in tauns:
        ws = np.logspace(-2, 2, 100)
        Sfs = [bv ** 2 + 2 * (bn * bv + bn ** 2) / (1 + (taun * w) ** 2) for w in ws]

        ax1.plot(ws, Sfs)

        data_file = home + "/CLionProjects/PhD/lif_green_noise/out/lif_green_noise_taua{:.2f}_delta{:.2f}_bv{:.2f}_bn{:.2f}_taun{:.2f}.dat".format(taua, delta, bv, bn, taun)
        isis = np.loadtxt(data_file)
        t_mean = np.mean(isis)
        CV = np.std(isis)/t_mean
        print(CV)
        k_correlations = []
        ks = np.arange(1, 6)
        std = fc.k_corr(isis, isis, 0)
        for k in ks:
            rhok = fc.k_corr(isis, isis, k)
            k_correlations.append(rhok/std)

        # SCC FROM THEORY
        k_correlations_theory = []
        for k in ks:
            k_corr = LIF_gnoise_scc(t_mean, mu, taua, delta, taun, bn, bv, k)
            k_correlations_theory.append(k_corr)

        ax2.scatter(ks, k_correlations, label="Simulation $C_V = {:.2f}$".format(CV))
        ax2.plot(ks, k_correlations_theory, label="Theory")

    ax1.set_xscale("log")
    ax1.grid(which='major', alpha=0.8, linestyle="--")
    ax1.tick_params(direction='in', labelsize=utl.labelsize)
    ax1.set_ylabel(r"$S(\omega)$", rotation=0, fontsize=utl.fontsize)
    ax1.set_xlabel("$\omega$", fontsize=utl.fontsize)

    ax2.grid(which='major', alpha=0.8, linestyle="--")
    ax2.tick_params(direction='in', labelsize=utl.labelsize)

    ax2.set_ylabel(r"$\rho_k$", rotation=0, fontsize=utl.fontsize)
    ax2.set_xlabel("$k$", fontsize=utl.fontsize)
    #ax2.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/LIF/plots/scc_lif_green_noise_taua{:.2f}_delta{:.2f}_bv{:.2f}_bn{:.2f}_taun{:.2f}.pdf".format(taua, delta, bv, bn, taun))
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        for j in range(1):
            gamma = 1
            mu = 5
            delta_a = 2.
            tau_a = 2.
            bn = -0.117 #0.1565 * tau_n  # sigma^2  = D/tau, thus if sigma is to be constant D must be increased propotional to0  tau
            bv = 0.232 #0.0067
            plt_scc_lif(tau_a, mu, delta_a, bn, bv)

