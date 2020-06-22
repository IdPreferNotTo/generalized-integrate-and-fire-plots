import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import os

def theta_model_prc():

    home = os.path.expanduser('~')
    mu = 1.01
    D=0.1
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    data_x = []
    data_y = []
    data_y2 = []
    data_y3 = []
    for i in range(1, 20):
        tau_n = i*0.2
        D = 0.1
        sigma = D / tau_n
        data_file = home + "/Data/Theta/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5e}_Delta0.0_{:d}.txt".format(mu, tau_n,
                                                                                                             D, 0)
        data = np.loadtxt(data_file, max_rows=25_000)
        t_det = np.pi

        t, a, eta, chi, chi2 = np.transpose(data)
        delta_t = [(x - t_det) for x in t]

        data_x.append(tau_n)
        data_y.append(np.mean(delta_t))
        w = 2
        f_vT = w
        c = (f_vT/2.)*erfc(-(f_vT)/np.sqrt(2*sigma)) + np.power(2*np.pi*sigma, -1/2)*sigma*np.exp(-f_vT*f_vT/(2*sigma))
        #prefactor = sigma*w*erfc(-w/np.sqrt(2*sigma))/(2*c)
        nom1 = (sigma/w)*np.pi*tau_n*tau_n
        nom2 = (sigma/w)*np.power(tau_n, 5)*w*w*(1-np.exp(-t_det/tau_n))
        denom = (1 + tau_n*tau_n*w*w)

        data_y2.append(nom1/denom + nom2/(denom*denom))
        data_y3.append((1/2)*t_det*sigma*tau_n*tau_n/(1+(w*w - sigma/2)*tau_n*tau_n))

    ax.scatter(data_x, data_y)
    ax.tick_params(direction='in')
    ax.plot(data_x, data_y2, c="C7")
    ax.plot(data_x, data_y3, c="C3")
    ax.set_xlabel(r"$\tau_\eta$")
    ax.set_ylabel(r"$\langle \delta T \rangle$")
    plt.tight_layout()
    plt.savefig(home + "/Data/Theta/red/plots/theta_D0.1.pdf")

    plt.show()
    return 1

if __name__ == "__main__":
    theta_model_prc()