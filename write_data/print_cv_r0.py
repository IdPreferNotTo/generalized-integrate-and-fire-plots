import numpy as np
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


def k_correlation(data1, data2, k):
    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])


def weighted_autocorrelation(t_det, tau_a, tau_n, gamma, mu, delta, k, v_t):
    alpha = np.exp(-t_det / tau_a)
    a_det = delta / (1 - alpha)
    sigma = np.sqrt(0.1 / tau_n)
    p1 = (alpha * a_det * sigma * np.exp(-gamma * t_det)) / (tau_a * (mu - gamma * v_t - a_det + delta))
    if k == 0:
        c1 = np.power(2 * gamma * (gamma + 1 / tau_n), -1) * (np.exp(2 * gamma * t_det) - 1)
        c2 = np.power((gamma - 1 / tau_n) * (gamma + 1 / tau_n), -1) * (np.exp((gamma - 1 / tau_n) * t_det) - 1)
        p2 = 2 * (c1 - c2)
    else:
        p2 = (tau_n ** 2) * (np.exp(-k * t_det / tau_n)) * (np.exp(t_det * (gamma - 1 / tau_n)) - 1) * (
                np.exp(t_det * (gamma + 1 / tau_n)) - 1) / ((gamma * tau_n) ** 2 - 1)
    return p1 * p1 * p2


def get_colored_cv_steps(t_det, a, chi, tau_a, tau_n, gamma, mu, delta, v_t):
    alpha = np.exp(-t_det / tau_a)
    a_fix = delta / (1 - alpha)
    beta = np.exp(-t_det / tau_n)
    nu = (mu - a_fix) * np.exp(-gamma * t_det) / (mu - v_t * gamma - a_fix + delta)
    f1 = np.power(tau_a / (a_fix * alpha * t_det), 2)
    p1 = (alpha ** 2 - 2 * nu * alpha ** 2 + 1) / (1 - np.power(alpha * nu, 2))
    p2 = 2 * (alpha * nu - alpha) * (1 - nu * alpha ** 2) / ((1 - (alpha * nu) ** 2) * (1 - alpha*nu*beta))
    # p2 = ((np.power(alpha, 2) - 2*np.power(alpha, 2)*nu+1)*(2*alpha*nu)-2*alpha*(1-np.power(alpha*nu, 2)))/(1-np.power(alpha*nu, 2))
    return f1 * (p1 * k_correlation(chi, chi, 0) + p2 * k_correlation(chi, chi, 1))


def get_white_cv(t_det, tau_a, gamma, mu, delta, v_t, D):
    alpha = np.exp(-t_det / tau_a)
    a_fix = delta / (1 - alpha)
    nu = (mu - a_fix) * np.exp(-gamma * t_det) / (mu - v_t * gamma - a_fix + delta)
    integral = np.power(mu - gamma*v_t - a_fix + delta, -2.)*np.power(2*gamma, -1.)*(1 - np.exp(-2*gamma*t_det))
    return 2*D*(1+alpha**2 - 2*nu*alpha**2)/((1-(alpha*nu)**2)*t_det**2) * integral


def write_cv_r0(tau_a, tau_n, gamma, delta, v_t, D):
    home = os.path.expanduser('~')
    t_mean_list = []
    t_det_list = []
    cv_colored_theorie = []
    cv_white_theorie = []
    cv_simulation = []

    for file in os.listdir(home + "/Data/scc_firing_rate_data/delta{:d}_taun{:.1f}/".format(delta, tau_n)):
        if file.endswith(".txt"):
            mu = file.split("_")[3]
            mu = float(mu[2:])
            data_file = home + "/Data/scc_firing_rate_data/delta{:d}_taun{:.1f}/".format(delta, tau_n) + file
            data = np.loadtxt(data_file)
            t_det = get_t_det(data_file)
            t, a, chi, x = np.transpose(data)
            t_mean = sum(t) / len(t)
            delta_t = [x - t_mean for x in t]
            variance = k_correlation(delta_t, delta_t, 0)
            t_mean_list.append(t_mean)
            t_det_list.append(t_det)
            cv_simulation.append(variance / np.power(t_mean, 2))
            cv_white_theorie.append(get_white_cv(t_mean, tau_a, gamma, mu, delta, v_t, D))
            cv_colored_theorie.append(get_colored_cv_steps(t_mean, a, chi, tau_a, tau_n, gamma, mu, delta, v_t))

    data = [list(x) for x in zip(t_det_list, t_mean_list, cv_simulation, cv_white_theorie, cv_colored_theorie)]
    data.sort(key=lambda x: x[0])
    cv_file_path = home + "/Data/scc_firing_rate_data/cv_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.txt".format(tau_a,
                                                                                                               tau_n,
                                                                                                               delta, D)
    with open(cv_file_path, "w") as cv_file:
        cv_file.write("# t_det | t_mean | cv_colored_simulation | cv_white_theory | cv_colored_theory \n")
        for (t, t_mean, cv_sim, cv_white, cv_color) in data:
            cv_file.write("{:.6f} {:.6} {:.6f} {:.6f} {:.6f} \n".format(t, t_mean, cv_sim, cv_white, cv_color))


if __name__ == "__main__":
    for i in [0.1, 1, 10]:
        for j in [10]:
            tau_a = 10
            tau_n = i
            gamma = 1
            delta = j
            D = 0.10
            v_t = 1
            write_cv_r0(tau_a, tau_n, gamma, delta, v_t, D)
