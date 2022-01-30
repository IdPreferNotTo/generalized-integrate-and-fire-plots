import numpy as np
from typing import List
from scipy.integrate import quad

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def delta(a, b):
    """
    Classical delta function \delta(a-b)
    """
    if a == b:
        return 1
    else:
        return 0


def theta(a, b):
    if a < b:
        return 0
    else:
        return 1


def moments(xs, k):
    """
    Calculates the k-th moment of the sample data xs:
    1/n \sum^n (x - <x>)**k
    where n the the sample length.
    """
    moment = 0
    mu = np.mean(xs)
    for x in xs:
        moment += (x - mu)**k
    return moment/len(xs)


def gaussian_dist(xs, mean, std):
    ys = []
    for x in xs:
        y = 1/np.sqrt(2*np.pi*(std**2)) * np.exp(-((x - mean)**2)/(2*std**2))
        ys.append(y)
    return ys


def coarse_grain_list(l: List[float], f: float):
    """
    Create a coarse grained version of the original list where the elements of the new list
    are the mean of the previous list over some window that is determined by f.
    f determines how many elements are averaged l_new[i] = mean(l[f*(i):f*(i+1)])
    """
    l_new = []
    max = int(len(l)/f) - 1
    for i in range(max):
        mean = np.mean(l[f*i:f*(i+1)])
        l_new.append(mean)
    return l_new


def get_states(n, m):
    xs = np.empty(n + m)
    for i in range(n + m):
        if i < m:
            xs[i] = 0
        if i >= m:
            xs[i] = n + m - i
    return xs


def steady_states_theory_invert_M(r_ref, r_opn, r_cls, n, m):
    # M is the actual transition matrix
    M = np.zeros([n+m, n+m])
    for i in range(n+m):
            if i < m-1:
                M[i, i] = - n*r_ref
                M[i+1, i] = n*r_ref
            elif i == m-1:
                M[i, i] = -n*r_opn
                for k in range(i+1, n+m):
                    M[k, i] = r_opn
            else:
                M[i, i] = -r_cls
                M[(i+1)%(n+m), i] = r_cls
    # M does not have full rank. Wherefor some row has to be replaced to reflect the additional normalization
    M[-1] = np.ones(n+m)
    Minv = np.linalg.inv(M)
    inhomgeneity = np.zeros(n+m)
    inhomgeneity[-1] = 1
    p0s = Minv.dot(inhomgeneity)
    return p0s


def f_from_k_invert_M(k, r_ref, r_opn, r_cls, n, m):
    M = np.zeros([n + m, n + m])
    for i in range(n + m):
        if i < m - 1:
            M[i, i] = - n * r_ref
            M[i + 1, i] = n * r_ref
        elif i == m - 1:
            M[i, i] = -n * r_opn
            for j in range(i + 1, n + m):
                M[j, i] = r_opn
        else:
            M[i, i] = -r_cls
            M[(i + 1) % (n + m), i] = r_cls
    M[-1] = np.ones(n + m)

    Minv = np.linalg.inv(M)
    p0s = steady_states_theory_invert_M(r_ref, r_opn, r_cls, n, m)
    p0s[-1] = 0
    p0s = np.asarray(p0s)
    deltas = [delta(k, i) for i in range(n+m)]
    deltas[-1] = 0
    inhomgeneity = np.subtract(p0s, deltas)
    f_from_k = Minv.dot(inhomgeneity)
    return f_from_k


def mean_puff_single(x, n, m, IP3):
    r_opn = 0.13 * np.power(x / 0.33, 3) * ((1 + 0.33 ** 3) / (1 + x ** 3)) * np.power(IP3 / 1., 3) * ((1. + 1. ** 3) / (1. + IP3 ** 3))
    r_ref = 1.3 * np.power(x / 0.33, 3) * ((1 + 0.33 ** 3) / (1 + x ** 3)) * np.power(IP3 / 1., 3) * ((1 + 1 ** 3) / (1. + IP3 ** 3))
    r_cls = 50

    p0s = steady_states_theory_invert_M(r_ref, r_opn, r_cls, n, m)
    xs = get_states(n, m)
    mean = sum([x * p for x, p in zip(xs, p0s)])
    return mean


def means_puff(N, tau, j, n, m, IP3):
    fs = []
    cas = np.linspace(0.01, 1.00, 100)
    for ca in cas:
        f = -(ca - 0.33)/tau + j*N*mean_puff_single(ca, n, m, IP3)
        fs.append(f)
    return fs


def intensity_puff_single(x, n, m, IP3):
    r_opn = 0.13 * np.power(x / 0.33, 3) * ((1. + 0.33 ** 3) / (1. + x ** 3)) * np.power(IP3 / 1., 3) * ((1. + 1. ** 3) / (1. + IP3 ** 3))
    r_ref = 1.3 * np.power(x / 0.33, 3) * ((1. + 0.33 ** 3) / (1. + x ** 3)) * np.power(IP3 / 1., 3) * ((1. + 1. ** 3) / (1. + IP3 ** 3))
    r_cls = 50

    xs = get_states(n, m)
    idxs = [i for i in range(n+m)]
    p0s = steady_states_theory_invert_M(r_ref, r_opn, r_cls, n, m)

    D_theory = 0
    for k in idxs:
        sum_over_i = 0
        f_from_k_to = f_from_k_invert_M(k, r_ref, r_opn, r_cls, n, m)
        for i in idxs:
            sum_over_i += xs[i] * f_from_k_to[i]
        D_theory += xs[k] * p0s[k] * sum_over_i
    return D_theory


def d_func(x, j, N, n, m, IP3):
    if x == 0:
        return 0
    else:
        return np.power(j, 2) * N * intensity_puff_single(x, n, m, IP3)


def f_func(x, tau, j, N, n, m, IP3):
    if x == 0:
        return -(x - 0.33) / tau
    else:
        f = mean_puff_single(x, n, m, IP3)
        return -(x - 0.33) / tau + j * N * f


def g_func(x, tau, j, N, n, m, IP3):
    f = f_func(x, tau, j, N, n, m, IP3)
    d = d_func(x, j, N, n, m, IP3)
    return f/d


def h_func(x, tau, j, N, n, m, IP3):
    #dca = 0.0001
    #h = 0
    #ca = 0.33
    #while(ca <= x):
    #    print(ca)
    #    g = g_func(ca, tau, j, N, n, m, IP3)
    #    h += g*dca
    #    ca += dca
    h = quad(g_func, 0.33, x, args=(tau, j, N, n, m, IP3))[0]
    return h


def firing_rate_no_adap(tau, j, N, n, m, IP3 = 1):
    cas_theory = np.linspace(0.30, 1, 10*700 + 1)
    dca = cas_theory[1] - cas_theory[0]
    p0s_theo_ca = []
    integral = 0

    for ca in reversed(cas_theory[1:]):
        print(f"{ca:.3f}")
        h = h_func(ca, tau, j, N, n, m, IP3)
        d = d_func(ca, j, N, n, m, IP3)
        if ca == 1:
            integral += 0
        elif ca >= 0.33:
            integral += np.exp(-h)*dca
        p0s_theo_ca.append(integral * np.exp(h) / d)
    print(p0s_theo_ca)
    norm = np.sum(p0s_theo_ca) * dca
    r0 = 1 / norm
    return r0

def k_corr(data1, data2, k):
    # Get two arbitrary data set and calculate their correlation with lag k.
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    data1 = [x - mu1 for x in data1]
    data2 = [x - mu2 for x in data2]
    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])


def fourier_transformation_isis(w, isis):
    t = 0
    f = 0
    for isi in isis:
        t += isi
        f += np.exp(1j*w*t)
    return f

def power_spectrum_isis(ws, isis, Tmax=2000):
    ISIs_chunks = []
    chunks = []
    t = 0
    for isi in isis:
        t += isi
        chunks.append(isi)
        if t > Tmax:
            ISIs_chunks.append(chunks.copy())
            chunks.clear()
            t = 0
    spectrum = []
    for w in ws:
        fws_real = []
        fws_img = []
        for isis in ISIs_chunks:
            fw = fourier_transformation_isis(w, isis)
            fws_real.append(fw.real)
            fws_img.append(fw.imag)
        spectrum.append((1. / Tmax) * (np.var(fws_real) + np.var(fws_img)))
    return spectrum


def inverse_gaussian(T, CV):
    ps = []
    ts = np.linspace(0, 2*T, 500)
    for t in ts:
        p = np.sqrt(T / (2 * np.pi * (CV**2) * (t ** 3))) * np.exp(-(t - T) ** 2 / (2 * T * (CV**2) * t))
        ps.append(p)
    return ts, ps