import numpy as np
import matplotlib.pyplot as plt
import os
import cmath
from utilites import functions as fc

def adap_cnoise_QIF_scc(t_det, prc, tau_a, tau_n, delta, k):
    alpha = np.exp(-t_det / tau_a)
    beta = np.exp(-t_det/tau_n) if tau_n != 0 else 0
    a_det = delta / (1 - alpha)

    zi = 0
    dt = t_det/len(prc)
    for n in range(len(prc)):
        zi += prc[n]*np.exp(-n*dt/tau_a)*dt
    nu = 1. - (a_det/tau_a)*zi

    if tau_n==0:
       chi2 = 0
       chi1 = 1
    else:
        chi2 = 0
        chi1 = 0
        dt = t_det / 100
        for n in range(100):
            for m in range(100):
                chi2 += prc[n] * prc[m] * np.exp(-(n - m)*dt / tau_n) * dt * dt
                chi1 += prc[n] * prc[m] * np.exp(-abs(n - m)*dt / tau_n) * dt * dt

    c1 = (1. + (alpha * nu) ** 2 - 2*alpha * nu * beta) / (alpha * nu - beta)
    c2 = (1. - (alpha * nu) ** 2) * (1 - alpha * beta) * (alpha - beta) / (
            (alpha * nu - beta) * (1. + alpha ** 2 - 2 * nu * alpha ** 2))
    c3 = 1. - alpha * nu * beta
    A = lambda i: alpha*(1 - alpha*alpha*nu)*(1-nu)*np.real(cmath.exp((i-1) * cmath.log(alpha*nu)))
    C = 1+ alpha**2 - 2 * alpha**2 * nu

    pa = lambda i : -A(i)/C
    pn = lambda i: beta**(i)*chi2/chi1

    factor1 = ((c1*pn(1) +c3)/(2*pa(1)*pn(1) + c3))
    factor2 = (c2/(2*pa(1)*pn(1) + c3))

    scc = []
    for n in range(1, k+1):
        scc.append(factor1*pa(n) + factor2*pn(n))
    return scc


def qif_prc(t_det, a_det, mu, tau_a):
    dt = 10**(-5)
    epsilon = 0.01
    ts=np.linspace(0, t_det, 100)
    prc = []
    for tk in ts:
        kick=True
        v = -10_000
        a = a_det
        t = 0
        while v < 10_000:
            if (t>=tk and kick==True):
                v = v+epsilon
                kick=False
            v += (mu + v**2 - a)*dt
            a += (-a/tau_a)*dt
            t += dt
        prc.append(-(t - t_det)/epsilon)
    return prc


def calculate_qif_timeseries(beta, tau_a, delta):
    vt = []
    at = []

    v = 0
    a = 0
    t = 0

    dt = 10**(-5)
    a_tmp0 = -1
    a_tmp1 = 0

    osc = 0
    count = 0
    while abs(a_tmp0 - a_tmp1) > 0.001:
        if v < 10_000:
            v += (beta + v**2  - a)*dt
            a += (-a/tau_a)*dt
        else:
            v = -10_000
            a += delta
            a_tmp1 = a_tmp0
            a_tmp0 = a
    while v < 10_000:
        if count%100==0:
            vt.append(v)
            at.append(a)
        v += (beta + v ** 2 - a) * dt
        a += (-a / tau_a) * dt
        t += dt
        count += 1
    vt.append(10_000)
    at.append(a)
    a += delta
    vt.append(10_000)
    at.append(a)
    v = -10_000
    vt.append(v)
    at.append(a)
    osc += 1
    return vt, at, a, t


def plt_scc_qif(tau_a, delta, mu, D):
    home = os.path.expanduser('~')
    #data_file = home + "/Data/QIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(mu, tau_a, tau_n, D, delta, 0)

    #print(data_file)
    #data = np.loadtxt(data_file)
    #t, a, eta, chi, chi2 = np.transpose(data)
    #t_mean = np.mean(t)
    #delta_t = [x - t_mean for x in t]

    # Get Data:
    vt, at, a_det, t_det = calculate_qif_timeseries(mu, tau_a, delta)
    prc = qif_prc(t_det, a_det, mu, tau_a)

    corr_sim = []
    #variance_delta_t = fc.k_corr(delta_t, delta_t, 0)
    k_range = range(1, 7)
    #for k in k_range:
    #    covariance_delta_t = fc.k_corr(delta_t, delta_t, k)
    #    corr_sim.append(covariance_delta_t / variance_delta_t)
    corr_theory_all = []
    for tau_n in [3, 4.5, 5]:
        corr_theory = adap_cnoise_QIF_scc(t_det, prc, tau_a, tau_n, delta, max(k_range))
        corr_theory_all.append(corr_theory)
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    #ax.scatter(k_range, corr_sim, s=20, ec="k", label="Simulation")
    for i in range(3):
        ax.plot(k_range, corr_theory_all[i], c="C{:d}".format(i), ls="--", lw=1, label="Theory")
    ax.set_xticks(k_range)
    ax.set_ylabel(r"$\rho_k$")
    ax.set_xlabel("$k$")
    #ax.legend(prop={"size":7})
    plt.tight_layout()
    #plt.savefig(home + "/Data/QIF/plots/scc_full_mu{:.1f}_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.pdf".format(mu, tau_a, tau_n, delta, D))
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        for j in range(1):
            gamma = 1.
            mu = 5
            v_t = 1
            tau_a = 6
            delta = 3
            D = 1.
            plt_scc_qif(tau_a, delta, mu, D)

