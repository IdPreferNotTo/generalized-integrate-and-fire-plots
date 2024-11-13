import numpy as np
import os
import matplotlib.pyplot as plt
from utilites import functions as fc


def calculate_covariance(k, alpha, beta, nu, variance, cross_correlation):
    x = variance*np.power(alpha*nu, k) + cross_correlation*np.power(alpha*nu, k-1)*(1 - np.power(beta/(alpha*nu), k))/(1-(beta/(alpha*nu)))
    return x


def plt_adaptation_covariance(t_det, gamma, mu, D, tau_a, tau_n, delta):
    home = os.path.expanduser('~')
    data_file = home + "/Data/cLIF/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.2f}.txt".format(mu, tau_a, tau_n, D)
    data = np.loadtxt(data_file)
    t, delta_a, chi, eta = np.transpose(data)
    covariance = []
    covariance_theory = []
    variance = fc.k_corr(delta_a, delta_a, 0)
    cross_correlation = fc.k_corr(delta_a, chi, 0)

    v_t = 1
    alpha = np.exp(-t_det/tau_a)
    beta = np.exp(-t_det/tau_n)
    a_fix = delta/(1-alpha)
    nu = (mu - a_fix) * np.exp(-gamma * t_det) / (mu - v_t * gamma - a_fix + delta)


    for k in range(11):
        covariance.append([k, fc.k_corr(delta_a, delta_a, k)])
        covariance_theory.append([k, calculate_covariance(k, alpha, beta, nu, variance, cross_correlation)])

    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    ax.scatter([x[0] for x in covariance], [x[1] for x in covariance], label = "simulation")
    ax.plot([x[0] for x in covariance_theory], [x[1] for x in covariance_theory], label = "theory", c= 'k')
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\langle \delta a_i \delta a_{i+k}\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(home + "/Data/cLIF/plots/covariance_a_taua{:.1f}_mu{:.1f}.pdf".format(tau_a, mu), transparent=True)
    plt.show()
    return 1

if __name__ == "__main__":
    t_det = 0.0251571
    gamma = 1
    mu = 80
    D = 0.1
    tau_a = 1
    tau_n = 1
    delta = 1
    plt_adaptation_covariance(t_det, gamma, mu, D, tau_a, tau_n, delta)