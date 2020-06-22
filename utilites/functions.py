import numpy as np
from scipy import integrate
import cmath

def get_lif_t_a_det(mu, tau_a, delta):
    v = 0
    a = 0
    t = 0
    t_det = 0
    dt = 10**(-5)
    a_tmp0 = -1
    a_tmp1 = 0
    while abs(a_tmp0 - a_tmp1) > 0.0001:
        if v < 1.:
            v += (mu - v - a) * dt
            a += (-a / tau_a) * dt
            t += dt
        else:
            t_det = t
            a_tmp1 = a_tmp0
            a_tmp0 = a

            v = 0
            a += delta/tau_a
            t = 0
    return t_det, a


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


def estimate_scc_error(lst, k, n):
    values = []
    m = int(len(lst) / n)
    lst_chunks = [lst[i:i + m] for i in range(0, len(lst), m)]
    for chunk in lst_chunks:
        values.append(k_corr(chunk, chunk, k) / k_corr(chunk, chunk, 0))
    var = np.var(values)
    error = 3 * np.sqrt(var / n)
    return error


def read_t_det(data_file):
    with open(data_file, "r") as data:
        for i, line in enumerate(data):
            if i == 1:
                t_det_str = line
            elif i > 2:
                break
    t_det = float(t_det_str[8:])
    return t_det


def read_w_det(data_file):
    with open(data_file, "r") as data:
        for i, line in enumerate(data):
            if i == 2:
                w_det_str = line
            elif i > 3:
                break
    w_det = float(w_det_str[8:])
    return w_det


def read_a_det(data_file):
    with open(data_file, "r") as data:
        for i, line in enumerate(data):
            if i == 3:
                a_det_str = line
            elif i > 4:
                break
    a_det = float(a_det_str[8:])
    return a_det


def k_corr(data1, data2, k):
    # Get two arbitrary data set and calculate their correlation with lag k.
    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])


def gif_varprc(t, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif):
    nu = 1. / tau_gif + gamma
    om02 = (beta_gif + gamma) / tau_gif
    A = nu ** 2 / 4 - om02
    if tau_a==0:
        ztt = 1./(mu - gamma * 1 - beta_gif*w_det)
    else:
        ztt = 1. / (mu - gamma * 1 - beta_gif * w_det - a_det + delta/tau_a)
    if A < 0:
        Om = np.sqrt(abs(A))
        return ztt * np.exp(0.5 * nu * (t - t_det)) * (
                np.cos(Om * (t - t_det)) + (gamma - 1. / tau_gif) / (2 * Om) * np.sin(Om * (t - t_det)))
    else:
        Om = np.sqrt(abs(A))
        return ztt * np.exp(0.5 * nu * (t - t_det)) * (
                np.cosh(Om * (t - t_det)) + (gamma - 1. / tau_gif) / (2 * Om) * np.sinh(Om * (t - t_det)))


def lif_varprc(t, t_det, a_det, mu, tau_a, delta):
    if tau_a == 0:
        return np.exp((t - t_det)) / (mu - 1)
    else:
        return np.exp((t - t_det)) / (mu - 1 - a_det + delta / tau_a)


def qif_varprc(t, mu):
    varprc = (1 / mu) * np.sin(2 * np.sqrt(mu) * t) ** 2
    return varprc


def cnoise_QIF_scc(t_det, tau_n, mu, k):
    chi2 = 0
    chi1 = 0
    ts = np.linspace(0, t_det, 100)
    dt = t_det / 100
    for t1 in ts:
        for t2 in ts:
            chi2 += qif_varprc(t1, mu) * qif_varprc(t2, mu) * np.exp(-abs(t_det + t2 - t1) / tau_n) * dt * dt
            chi1 += qif_varprc(t1, mu) * qif_varprc(t2, mu) * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
    beta = np.exp(-t_det / tau_n)
    pn = lambda i: beta ** (i - 1) * chi2 / chi1
    print(r"$\rho_{\eta,1}$", pn(1))
    return pn(k)


def LIF_scc(t_det, mu, tau_a, delta, tau_n, Dn, Dw, k):
    if tau_a == 0:
        alpha = 0
        a_det = 0
        nu = 1
        def pa(i): return 0
    else:
        alpha = np.exp(-t_det / tau_a)
        beta = np.exp(-t_det / tau_n) if tau_n != 0 else 0
        a_det = delta / (tau_a * (1 - alpha))
        def h(t): return lif_varprc(t, t_det, a_det, mu, tau_a, delta) * np.exp(-t / tau_a)
        zi = integrate.quad(h, 0, t_det)[0]
        nu = 1. - (a_det / tau_a) * zi

        def a(i): return alpha * (1 - alpha * alpha * nu) * (1 - nu) * np.real(cmath.exp((i - 1) * cmath.log(alpha * nu)))
        c = 1 + alpha ** 2 - 2 * alpha ** 2 * nu
        def pa(i): return -a(i) / c

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
                chi2 += lif_varprc(t1, t_det, a_det, mu, tau_a, delta) * lif_varprc(t2, t_det, a_det, mu, tau_a,
                                                                                    delta) * np.exp(
                    -(t2 - t1) / tau_n) * dt * dt
                chi1 += lif_varprc(t1, t_det, a_det, mu, tau_a, delta) * lif_varprc(t2, t_det, a_det, mu, tau_a,
                                                                                    delta) * np.exp(
                    -abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            chi1 += (2*tau_n * Dw / Dn) * lif_varprc(t1, t_det, a_det, mu, tau_a, delta) ** 2 * dt

    def pn(i): return beta ** (i) * chi2 / chi1

    A = 1 + (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta)/((alpha * nu - beta)) * pn(1) - alpha * nu * beta
    B = (1. - (alpha * nu) ** 2) * (1. - alpha * beta)*((alpha - beta)) / ((1 + alpha ** 2 - 2 * alpha ** 2 * nu)*(alpha * nu - beta))
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta
    return (A/C) * pa(k) + (B/C) * pn(k)


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
            return gif_varprc(t, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-t / tau_a)

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
                chi2 += gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * gif_varprc(t2, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-abs(t_det + t2 - t1) / tau_n) * dt * dt
                chi1 += gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * gif_varprc(t2, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            chi1 += (2 * tau_n * Dw / Dn) * gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) ** 2 * dt

    def pn(i):
        return beta ** (i - 1) * chi2 / chi1

    A = (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta) * pn(1)
    B = 1 - alpha * nu * beta
    D = (1. - (alpha * nu) ** 2) * (1. - alpha * beta) / (1 + alpha ** 2 - 2 * alpha ** 2 * nu)
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta

    factor1 = (A / (alpha * nu - beta) + B) / C
    factor2 = (D / C) * (alpha - beta) / (alpha * nu - beta)
    return factor1 * pa(k) + factor2 * pn(k)