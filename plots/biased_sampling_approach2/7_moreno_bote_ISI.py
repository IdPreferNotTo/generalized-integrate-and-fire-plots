import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

def gaus(x, sigma, mean):
    return ((1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-mean)**2/(2*sigma)))

def transferfunction(n, mu, vR, vT):
    return 1/np.log((mu+n-vR)/(mu+n-vT))


def plt_moreno_bote(tau_n):
    home = os.path.expanduser('~')
    data_x = []
    data_y = []
    theory_data_x = []
    theory_data_y1 = []
    theory_data_y2 = []
    theory_data_y3 = []


    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    for i in np.logspace(-1, 2, 25):
        mu = 1.1
        D = i*((mu-1)**2)*tau_n
        sigma = D/tau_n
        print(sigma)
        gamma = 1
        vR = 0
        vT = 1
        f_vT = mu - gamma * vT
        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5e}_Delta0.0_{:d}.txt".format(mu,tau_n,D, 0)
        print(data_file)
        t_det = np.log((mu - vR) / (mu - vT))
        data = np.loadtxt(data_file, max_rows = 25_000)
        t, a, eta, chi, chi2 = np.transpose(data)
        delta_t = [(x - t_det) / t_det for x in t]
        data_y.append(np.mean(delta_t))
        data_x.append(sigma)

        r0 = np.power(np.log((vR - mu) / (vT - mu)), -1)
        fR = np.power((vR - mu), 1)
        fT = np.power((vT - mu), 1)
        r1 = (sigma) * np.power(r0, 2.) * (
                    r0 * np.power(1/fT - 1/fR, 2.) - ((1/(fT**2) - 1/(fR**2)) / 2.))
        if (sigma == 0):
            r = [r0, 0]
        else:
            r = integrate.quad(lambda n: gaus(n, sigma, 0) * transferfunction(n, mu, vR, vT), -f_vT, np.infty)

        theory_data_x.append(sigma)
        theory_data_y1.append(0)
        #theory_data_y2.append(r0/(r0+r1) -1)
        theory_data_y3.append(r0/r[0] -1)

    #ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma^2 / f(v_T)^2$")
    ax.set_ylabel(r"$\langle \delta T \rangle/T$")
    ax.tick_params(direction='in')
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.scatter(data_x, data_y)
    ax.plot(theory_data_x, theory_data_y1, c="C3", label=r"$T=$ {:.2f}".format(t_det))
    #ax.plot(theory_data_x, theory_data_y2, c="C4")
    ax.plot(theory_data_x, theory_data_y3, c="k", label=r"$\tau_\eta=${:.2f}".format(tau_n))
    ax.legend()
    plt.show()

    return 1

if __name__ == "__main__":
    for i in range(6):
        tau_n = [0.05, 0.1, 0.5, 1., 5., 10.][i]
        plt_moreno_bote(tau_n)