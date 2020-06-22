import numpy as np
import matplotlib.pyplot as plt


def lif_perturbation(mu, tau_a, delta, t_kick, e_kick):
    v = 0
    a = 0
    t = 0
    dt = 10**(-4)
    a_tmp0 = -1
    a_tmp1 = 0
    t_det = 0
    while abs(a_tmp0 - a_tmp1) > 0.001:
        while v < 1.:
            v += (mu - v - a) * dt
            a += (-a / tau_a) * dt
            t += dt
            t_det = t
        print("fire")
        a_tmp1 = a_tmp0
        a_tmp0 = a
        v = 0
        a += delta/tau_a
        t = 0

    vt1 = []
    vt2 = []
    at = []
    tt = []
    v1 = 0
    v2 = 0
    while v1 < 1.:
        if t < t_det*t_kick:
            v1 += (mu - v1 - a) * dt
            v2 += (mu - v2 - a) * dt
            a += (-a / tau_a) * dt
            t += dt

            vt1.append(v1)
            vt2.append(v2)
            at.append(a)
            tt.append(t)
        else:
            v2 += e_kick
            t_kick += 2
    return vt1, vt2, at, tt


def plot_perturbation():
    vt1, vt2, at, tt = lif_perturbation(2, 1, 1, 0.1, 0.1)
    plt.plot(tt, vt1)
    plt.plot(tt, vt2)
    plt.show()

if __name__ == "__main__":
    plot_perturbation()