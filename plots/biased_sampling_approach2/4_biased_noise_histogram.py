import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erf
import scipy.integrate as integrate
import os


def plt_ISI_histogram():
    home = os.path.expanduser('~')

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2)
    ax = fig.add_subplot(gs[0:3,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[2,1])
    axis = [ax1, ax2, ax3]
    ax.tick_params(direction='in')
    for a in axis:
        a.tick_params(direction='in')
    colors = ["C0", "C1", "C2"]
    mu = 2.0
    D = 0.4
    for i in range(3):
        tau_n = [0.1, 1, 3.][i]
        vR = 0
        vT = 1
        sigma = D/tau_n
        data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_{:d}.txt".format(mu, tau_n,
                                                                                                                 D, 0)
        print(data_file)
        data = np.loadtxt(data_file)
        t, a, chi, eta, chi2 = np.transpose(data)

        f_vT = mu-1.
        c = (f_vT/2)*erfc(-(f_vT)/np.sqrt(2*sigma)) + np.power(2*np.pi*sigma, -1/2)*sigma*np.exp(-f_vT*f_vT/(2*sigma))
        mean = sigma*(erfc(-f_vT/np.sqrt(2*sigma)))/(2*c)
        print(mean)
        xs = np.linspace(-5*(i+1)*sigma, 5*(i+1)*sigma, num=100)
        pxs = [np.power(2*sigma*np.pi, -1/2)*(1/c)*(f_vT + x)*np.exp(-x*x/(2*sigma)) for x in xs]
        xpxs = [x*px for x, px in zip(xs, pxs) if x > -1]
        if(i==0):
            ax.plot(xs, pxs, c="C3", label="Theory")
        else:
            ax.plot(xs, pxs, c="C3")
        ax.hist(eta, bins = 100, density=True, alpha = 0.5, label = r"$\tau_\eta$ = {:.1f}".format(tau_n))
        axis[i].plot(xs, pxs, c="C3", label = "{:.1e}".format(mean))
        axis[i].hist(eta, bins = 100, density=True, alpha = 0.5, color=colors[i], label = "{:.1e}".format(np.mean(eta)))
        axis[i].legend()
        if(i==2):
            axis[i].set_xlabel(r"$\eta(t_i)$")
        ax.legend()
    ax.set_xlabel(r"$\eta(t_i)$")
    plt.savefig(home + "/Data/LIF/red/plots/noise_upon_firing_mu{:.2f}_D{:.2f}.pdf".format(mu, D))
    plt.show()
    return 1


if __name__ == "__main__":
    plt_ISI_histogram()