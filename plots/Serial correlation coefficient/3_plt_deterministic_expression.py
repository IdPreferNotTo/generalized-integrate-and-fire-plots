import numpy as np
import matplotlib.pyplot as plt
import os


def get_t_det(data_file):
    with open(data_file, "r") as data:
        for i, line in enumerate(data):
            if i == 2:
                t_det_str = line
            elif i > 3:
                break
    t_det = float(t_det_str[8:])
    return t_det

def analytic_magic(t_det, chi, tau_a, tau_n, gamma, mu, delta, v_t):
    alpha = np.exp(-t_det/tau_a)
    a_fix = delta/(1-alpha)
    nu = (mu - a_fix)*np.exp(-gamma*t_det)/(mu - v_t*gamma - a_fix + delta)
    beta = np.exp(-t_det/tau_n)
    p1 = 2*alpha*nu/(1-np.power(alpha*nu, 2))
    p2 = (1-alpha*nu*beta)/(1-np.power(alpha*nu, 2))
    return p1 + p2*(k_correlation(chi, chi, 0)/k_correlation(chi, chi, 1))


def k_correlation(data1, data2, k):
    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data1) #len(data2[k:])


def could_it_be(mu, tau_a, gamma, delta, D):
    v_t = 1

    home = os.path.expanduser('~')
    tau_n_list  = []
    correlations = []
    correlations_theory = []
    for i in range(20):
        tau_n = 0.1 * (i + 1)
        data_file = home + "/Data/cLIF/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.2f}.txt".format(mu, tau_a, tau_n, D)
        t_det = get_t_det(data_file)
        data = np.loadtxt(data_file)
        t, a, chi, eta = np.transpose(data)
        tau_n_list.append(tau_n)
        correlations.append(k_correlation(a, chi,0)/k_correlation(a,a,0))
        correlations_theory.append(analytic_magic(t_det, chi, tau_a, tau_n, gamma, mu, delta, v_t))

    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.scatter(tau_n_list, correlations, s=10, c='k', label="simulation")
    ax.plot(tau_n_list, [1/x for x in correlations_theory], label="theory")
    ax.grid(which='major', alpha=0.8, linestyle="--")
    ax.tick_params(direction='in')
    ax.set_xlabel(r"$\tau_\eta$")
    ax.set_ylabel(r"$\langle \delta a_i \chi_i \rangle/\langle \delta a_i^2\rangle $")
    plt.tight_layout()
    plt.savefig(home + "/Data/cLIF_plots/non_stochastic_exp_mu{0:.1f}_taua{1:.1f}_D{2:.1f}.pdf".format(mu, tau_a, D), transparent = True)

    plt.show()

    return 0

if __name__ == '__main__':
    mu = 80.0
    tau_a = 10
    gamma = 1
    delta = 1
    D = 0.10
    could_it_be(mu, tau_a, gamma, delta, D)