import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import os

def gaus(x, sigma, mean):
    return ((1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-mean)**2/(2*sigma)))

def transferfunction(n, mu, vR, vT):
    return 1/np.log((mu+n-vR)/(mu+n-vT))

def troubleshooting():
    home = os.path.expanduser('~')
    data_x = []
    data_y = []
    data_y2 = []
    f, axis = plt.subplots(3, 3, figsize=(3*3, 9*3 / 4), sharey=True, sharex=True)
    dots = np.logspace(-1., 2, num=25)
    for n, ax_row in enumerate(axis):
        for m, ax in enumerate(ax_row):
            ax.tick_params(direction='in')
            ax.set_xscale("log")
            if (m == 0):
                ax.set_ylabel(r"$\langle \delta T \rangle / T$")
            if (n==2):
                ax.set_xlabel(r"$\sigma^2 / f(v_T)^2$")

    for n, tau_n in enumerate([0.1, 1, 10]) :
        for m, mu in  enumerate([1.01, 1.1, 1.5]):
            for i in dots:
                gamma = 1
                vR = 0
                vT = 1
                f_vT = mu - gamma * vT

                sigma = (i)*f_vT**2
                D = sigma*tau_n

                data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5e}_Delta0.0_{:d}.txt".format(mu,tau_n,D, 0)
                print(data_file)
                data = np.loadtxt(data_file, max_rows=25_000)
                t, a, eta, chi, chi2 = np.transpose(data)
                t_det = np.log((mu - vR) / (mu - vT))
                delta_t = [(x - t_det) / t_det for x in t]

                data_y2.append(np.mean(delta_t))

                r0 = 1/np.log(mu/(f_vT))
                if(sigma==0):
                    result = [transferfunction(0, mu, vR, vT), 0]
                else:
                    result = integrate.quad(lambda n: gaus(n, sigma, 0)*transferfunction(n, mu, vR, vT), -f_vT, np.infty)
                data_x.append(sigma/((f_vT)**2))
                data_y.append(r0/result[0] -1)

            axis[n][m].plot(data_x, data_y, label="$\mu =$ {:.2f} \n".format(mu) + r"$\tau_\eta=$ {:.1f}".format(tau_n))
            axis[n][m].scatter(data_x, data_y2, c="k")
            axis[n][m].legend()
            data_x = []
            data_y = []
            data_y2 = []

    plt.savefig(home + "/Data/LIF/red/plots/delta_t_moreno_bote_theory.pdf")
    plt.show()
    return 1


if __name__ == "__main__":
    troubleshooting()