import numpy as np
import os

def calculate_qif_noisy_timeseries(beta, tau_a, delta, tau_n, D):
    vt = []
    at = []
    nt = []
    tt = []

    dt = 10**(-5)
    v = -10_000
    a = 0
    a_tmp0 = -1
    a_tmp1 = 0

    while abs(a_tmp0 - a_tmp1) > 0.001:
        if v < 10_000:
            v += (beta + v**2  - a)*dt
            a += (-a/tau_a)*dt
        else:
            v = -10_000
            a += delta
            a_tmp1 = a_tmp0
            a_tmp0 = a

    a_det = a
    v = -10_000
    a = a_det
    n = 0
    t = 0
    osc = 0
    count = 0
    while osc < 20:
        print(osc)
        while v < 10_000:
            xi = np.random.normal(loc=0, scale=1, size=1)
            if count%100==0:
                vt.append(float(v))
                at.append(float(a))
                nt.append(float(n))
                tt.append(float(t))
            v += (beta + v ** 2 - a + n) * dt
            a += (-a / tau_a) * dt
            n += (-n/tau_n)*dt + (np.sqrt(2*D*dt)/tau_n)*xi
            t += dt
            count += 1
        vt.append(10_000)
        at.append(float(a))
        nt.append(float(n))
        tt.append(float(t))
        a += delta
        vt.append(10_000)
        at.append(float(a))
        tt.append(float(t))
        v = -10_000
        vt.append(float(v))
        at.append(float(a))
        tt.append(float(t))
        osc +=1
    home = os.path.expanduser('~')
    path = home + "/Data/QIF/data_full/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
        mu, tau_a, tau_n, D, delta, 0)

    with open(path, "w") as file:
        file.write("#mu: {:.2f}, tau_n: {:.2f}, tau_a: {:.2f}, delta: {:.2f} \n".format(mu, tau_n, tau_a, delta))
        file.write("# t v a n \n")
        t_det = -tau_a*np.log(1. - delta/a_det)
        file.write("#tDet: {:.5f}\n".format(t_det))
        file.write("#wDet: {:.5f}\n".format(0))
        file.write("#aDet: {:.5f}\n".format(a_det))
        file.write("#stepSize: {:.0e}\n".format(dt))
        for v, a, n, t in zip(vt, at, nt, tt):
            file.write("{:.5f} {:.5f} {:.5f} {:.5f}\n".format(t, v, a, n))
    return vt, at

if __name__ == "__main__":
    for m in range(1):
        for k in range(1):
            for i in range(1):
                mu = 5
                tau_a = [5, 10][m]
                tau_n = [4.][k]
                delta = [3, 1][m]
                D = 1.0
                calculate_qif_noisy_timeseries(mu, tau_a, delta, tau_n, D)