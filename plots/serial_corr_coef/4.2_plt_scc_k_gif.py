import numpy as np
import matplotlib.pyplot as plt
import os
from utilites import functions as fc
from scipy import integrate
import cmath

def get_gif_t_a_w_det(gamma, mu, beta, tau_w, tau_a, delta):
    v = 0
    w = 0
    a = delta
    t = 0
    t_det = 0
    w_det = 0
    dt = 10 ** (-5)
    a_tmp0 = -1
    a_tmp1 = 0
    while abs(a_tmp0 - a_tmp1) > 0.0001:
        if v < 1.:
            v_tmp = v
            v += (mu - gamma*v - beta * w - a) * dt
            w += ((v_tmp - w) / tau_w) * dt
            a += (-a / tau_a) * dt
            t += dt
        else:
            t_det = t
            w_det = w
            a_tmp1 = a_tmp0
            a_tmp0 = a
            v = 0
            w = 0
            a += delta/tau_a
            t = 0
    return t_det, w_det, a

def GIF_scc(t_det, w_det, gamma, mu, tau_gif, beta_gif, tau_a, delta, tau_n, Dn, Dw, k):
    if tau_a == 0:
        alpha = 0
        a_det = 0
        nu = 1
        def pa(i): return 0
    else:
        alpha = np.exp(-t_det / tau_a)
        a_det = delta / (tau_a*(1. - alpha))

        def h(t):
            return fc.gif_varprc(t, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-t / tau_a)

        zi = integrate.quad(h, 0, t_det)[0]
        nu = 1. - (a_det / tau_a) * zi

        def a(i):
            return alpha * (1 - alpha * alpha * nu) * (1 - nu) * np.real(cmath.exp((i - 1) * cmath.log(alpha * nu)))

        c = np.power(alpha, 2) - 2 * np.power(alpha, 2) * nu + 1

        def pa(i):
            return -a(i) / c

    if tau_n == 0:
        beta = 0
        chi2 = 0
        chi1 = 1
    else:
        beta = np.exp(-t_det / tau_n)
        chi2 = 0
        chi1 = 0
        ts = np.linspace(0, t_det, 100)
        dt = t_det / 100
        for t1 in ts:
            for t2 in ts:
                chi2 += fc.gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * fc.gif_varprc(t2, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-abs(t_det + t2 - t1) / tau_n) * dt * dt
                chi1 += fc.gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * fc.gif_varprc(t2, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            chi1 += (2 * tau_n * Dw / Dn) * fc.gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) ** 2 * dt

    def pn(i):
        return beta ** (i - 1) * chi2 / chi1

    A = (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta) * pn(1)
    B = 1 - alpha * nu * beta
    D = (1. - (alpha * nu) ** 2) * (1. - alpha * beta) / (1 + alpha ** 2 - 2 * alpha ** 2 * nu)
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta

    factor1 = (A / (alpha * nu - beta) + B) / C
    factor2 = (D / C) * (alpha - beta) / (alpha * nu - beta)
    return factor1 * pa(k) + factor2 * pn(k)

def plt_scc_GIF(gamma, mu, tau_a, tau_w, tau_n, beta, Dn, Dw, delta):
    home = os.path.expanduser('~')
    #data_file = home + "/Data/GIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio0.00.txt".format(mu, tau_a, tau_n, D, delta)

    #t_det = fc.read_t_det(data_file)
    #w_det = fc.read_w_det(data_file)
    #data = np.loadtxt(data_file)
    #t, a, eta, chi, chi2 = np.transpose(data)
    #t_mean = np.mean(t)
    #delta_t = [x - t_mean for x in t]

    #corr_adap_cnoise_sim = []
    corr_adap_theory = []
    t_det, w_det, a_det = get_gif_t_a_w_det(gamma, mu, beta, tau_w, tau_a, delta)
    print(t_det)
    #variance_delta_t = fc.k_corr(delta_t, delta_t, 0)
    k_range = range(1, 10)
    for k in k_range:
        #covariance_delta_t = fc.k_corr(delta_t, delta_t, k)
        #corr_adap_cnoise_sim.append(covariance_delta_t/variance_delta_t)
        corr_adap_theory.append(GIF_scc(t_det, w_det, gamma, mu, tau_w, beta, tau_a, delta, tau_n, Dn, Dw, k))
    f, ax  = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    #ax.scatter(k_range, corr_adap_cnoise_sim, label = "simulation")
    ax.plot(k_range, corr_adap_theory, c="k", ls='--', label = "{:.2f}".format(t_det))
    ax.set_ylabel(r"$\rho_k$")
    ax.set_xlabel("k")
    ax.legend()
    plt.tight_layout()
    #plt.savefig(home + "/Data/GIF/plots/scc_full_mu{:.1f}_taua{:.1f}_taun{:.2f}_delta{:.1f}_D{:.1f}.pdf".format(mu, tau_a, tau_n, delta, D), transparent=True)
    plt.show()


if __name__ == "__main__":
    for i in range(1):
        mu = 1
        gamma = -1
        betaw = 5
        tauw = 1.1
        taua = 1
        delta = 2.3
        taun = 20
        Dn = 0.1
        Dw = 0#0.01
        plt_scc_GIF(gamma, mu, taua, tauw, taun, betaw, Dn, Dw, delta)
