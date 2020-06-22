import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import os


def calculate_resp(vR, vT, mu, D, tau, gamma):
    sigma = D/tau
    f_vT = mu-vT
    t_det = np.log((mu - vR) / (mu - vT))
    p1 = -(np.exp(-t_det/tau) - np.exp(-gamma*t_det))/(gamma - 1/tau)
    p2 = -(1 - np.exp(-2*gamma*t_det))/(2*gamma + 2/tau)
    p3 = (gamma*(np.exp(-(gamma+1/tau)*t_det) - np.exp(-2*gamma*t_det)))/(gamma*gamma - 1/(tau*tau))
    p4 = -(np.exp(-gamma*t_det) - 1.)*(1-np.exp(-gamma*t_det)*np.exp(-t_det/tau))/(gamma + 1/tau)
    result = (sigma/np.power(f_vT,2))*(p2 + p3 + p4/2)
    return result

def calculate_fourier_resp(vR, vT, mu, D, tau, gamma):
    delta_w = 0
    sigma = D/tau
    f_vT = mu-vT
    t_det = np.log((mu - vR) / (mu - vT))
    w = 2*np.pi/t_det
    cn2 = lambda n: ((2./(t_det*f_vT))**2) * np.power(1 - np.exp(-t_det),2)/(1 + np.power(w*n, 2))
    for n in range(-100, 100):
        delta_w += (n*n*tau*tau*cn2(n))/(1 + (w*n*tau)**2)
    delta_w = w * (sigma / 2)*delta_w
    return delta_w



def troubleshooting():
    home = os.path.expanduser('~')
    data_x = []
    data_y = []
    data_y2 = []
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.tick_params(direction='in')
    #ax.set_xscale("log")
    ax.set_ylabel(r"$\langle \delta T \rangle / T$")
    ax.set_xlabel(r"$\tau_\eta$")

    #for n, tau_n in enumerate([0.1, 1, 10]) :
    for mu in ([1.1]):
        for i in range(1, 20):
            gamma = 1
            vR = 0
            vT = 1
            f_vT = mu - gamma * vT
            tau_n = 0.2*i
            sigma = 0.02
            D=sigma*tau_n

            data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5e}_Delta0.0_{:d}.txt".format(mu,tau_n,D, 0)
            print(data_file)
            data = np.loadtxt(data_file, max_rows=25_000)
            t, a, eta, chi, chi2 = np.transpose(data)
            part = int(len(t)/10)
            for i in range(10):
                np.mean(t[i*part:(i+1)*part])


            t_det = np.log((mu - vR) / (mu - vT))
            print(t_det)
            delta_t = [(x - t_det) for x in t]

            data_y2.append(np.mean(delta_t))
            data_x.append(tau_n)
            data_y.append(calculate_resp(vR, vT, mu, D, tau_n, 1))

        ax.plot(data_x, data_y)
        ax.scatter(data_x, data_y2, c="k", label="$\mu =$ {:.2f} \n".format(mu) + r"$\sigma^2/f(v_T)^2=$ {:.2f}".format(sigma/f_vT**2))
        ax.legend()

    #plt.savefig(home + "/Data/LIF/red/plots/delta_t_moreno_bote_theory.pdf")
    plt.show()
    return 1


if __name__ == "__main__":
    troubleshooting()