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
        c1 = np.power(2*gamma*(gamma + 1/tau_n), -1)*(np.exp(2*gamma*t_det) - 1)
        c2 = np.power((gamma - 1/tau_n)*(gamma + 1/tau_n), -1)*(np.exp((gamma-1/tau_n)*t_det) -1)
        p2 = 2*(c1 - c2)
    else:
        p2 = (tau_n ** 2) * (np.exp(-k*t_det / tau_n)) * (np.exp(t_det * (gamma - 1 / tau_n)) - 1) * (
            np.exp(t_det * (gamma + 1 / tau_n)) - 1) / ((gamma * tau_n) ** 2 - 1)
    return p1 * p1 * p2

def get_colored_correlation(t_det, a, chi, tau_a, tau_n, gamma, mu, delta, v_t, k):
    alpha = np.exp(-t_det / tau_a)
    beta = np.exp(-t_det/tau_n)
    a_fix = delta / (1 - alpha)
    nu = (mu - a_fix) * np.exp(-gamma * t_det) / (mu - v_t * gamma - a_fix + delta)
    x = beta/(alpha*nu)
    chi0 = weighted_autocorrelation(t_det, tau_a, tau_n, gamma, mu, delta, 0, v_t)
    chi1 = weighted_autocorrelation(t_det, tau_a, tau_n, gamma, mu, delta, 1, v_t)
    D = 2*alpha*nu/(1-np.power(alpha*nu, 2)) + ((1-alpha*nu*beta)/(1-np.power(alpha*nu, 2)))*chi0/chi1
    A_k = alpha*(np.power(alpha, 2)*nu -1)*(1-nu)*np.power(alpha*nu, k-1)
    B_k = np.power(alpha*nu*(1-x), -1)*(A_k - (alpha*beta -1)*(alpha-beta)*np.power(beta, k-1))
    C = np.power(alpha, 2) - 2*np.power(alpha, 2)*nu + 1
    nominator = A_k*D + B_k
    denominator = C*D - 2*alpha
    rho_k = nominator/denominator
    return rho_k

def get_gaussian_correlation(t_det, tau_a, gamma, mu, delta, v_t, k):
    alpha = np.exp(-t_det / tau_a)
    a_fix = delta / (1 - alpha)
    nu = (mu - a_fix) * np.exp(-gamma * t_det) / (mu - v_t * gamma - a_fix + delta)
    A_k = -alpha*(1-(alpha**2)*nu)*(1-nu)*(alpha*nu)**(k-1)
    C = 1 + np.power(alpha, 2) - 2*np.power(alpha, 2)*nu
    return  (A_k/C)


def write_scc_firing_rate(k, tau_a, tau_n, gamma, delta, v_t):
    home = os.path.expanduser('~')
    t_det_list = []
    corr_colored_theorie = []
    corr_white_theorie = []
    corr_simulation = []

    tau_n = tau_n + 0.01
    for file in os.listdir(home + "/Data/scc_firing_rate_data/delta{:d}_taun{:.0f}/".format(delta, tau_n)):
        if file.endswith(".txt"):
            mu = file.split("_")[3]
            mu = float(mu[2:])
            data_file = home + "/Data/scc_firing_rate_data/delta{:d}_taun{:.0f}/".format(delta, tau_n) + file
            data = np.loadtxt(data_file)
            #t_det = get_t_det(data_file)
            t, a, chi, x = np.transpose(data)
            t_det = sum(t)/len(t)
            delta_t = [x - t_det for x in t]
            variance = k_correlation(delta_t, delta_t, 0)
            covariance = k_correlation(delta_t, delta_t, k)
            t_det_list.append(t_det)
            corr_simulation.append(covariance / variance)
            corr_colored_theorie.append(get_colored_correlation(t_det, a, chi, tau_a, tau_n, gamma, mu, delta, v_t, k))
            corr_white_theorie.append(get_gaussian_correlation(t_det, tau_a, gamma, mu, delta, v_t, k))

    data = [list(x) for x in zip(t_det_list, corr_simulation, corr_white_theorie, corr_colored_theorie)]
    data.sort(key = lambda x: x[0])
    scc_file_path = home + "/Data/scc_firing_rate_data/scc{:d}_taua{:.1f}_taun{:.1f}_delta{:.1f}_D{:.1f}.txt".format(k, tau_a, tau_n, delta, D)
    with open(scc_file_path, "w") as scc_file:
        scc_file.write("# t_det | corr_colored_simulation | corr_white_theory | corr_colored_theory \n")
        for (t, corr_sim, corr_white, corr_color) in data:
            scc_file.write("{:.6f} {:.6f} {:.6f} {:.6f} \n".format(t, corr_sim, corr_white, corr_color))

if __name__ == "__main__":
    k = 2
    tau_a = 10
    tau_n = 1
    gamma = 1
    delta = 1
    D = 0.10
    v_t = 1
    write_scc_firing_rate(k, tau_a, tau_n, gamma, delta, v_t)

