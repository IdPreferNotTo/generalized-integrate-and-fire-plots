import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
import scipy.integrate as integrate
from scipy.special import erfc
import matplotlib.ticker as ticker
import os

def plt_mean_ISI_der_sigma():
    home = os.path.expanduser('~')

    mu = 2.0

    f, ax = plt.subplots(1, 1, figsize=(4, 9 / 3))
    #ax.annotate("$\sigma^2_\eta = f(v_t)^2$", xy=(np.power((mu - 1), 2) + 0.01, -0.03), rotation=90)
    ax.set_xlabel("$D$")
    ax.set_ylabel(r"$\langle \delta T \rangle/T$")
    ax.grid(which='major', alpha=0.8, linestyle="--")

    for i in range(4):
        Ds = []
        plot_data = []
        data_chi1 = []
        data_chi2 = []
        plot_error_data = []

        cv = False
        cv_line = 0

        vR = 0
        vT = 1
        for k in range(1, 10):
            tau_n = [0.0, 0.1, 1.0, 3.0][i]
            D = 0.05*k
            Ds.append(D)

            data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a0.0_tau_n{:.1f}_D{:.5f}_Delta0.0_{:d}.txt".format(mu, tau_n, D, 0)

            print(data_file)
            if(tau_n == 0.0): tau_n = 0.001
            sigma = D/tau_n
            t_det = np.log((mu-vR)/(mu-vT))
            data = np.loadtxt(data_file, max_rows=50_000)
            t, a, chi, eta, chi2 = np.transpose(data)
            delta_t = [(x - t_det)/t_det for x in t]
            plot_data.append(sum(delta_t)/len(delta_t))


            if(tau_n==1):tau_n=1.01
            lin_resp = sigma*np.exp(-t_det)*(np.exp((1-1/tau_n)*t_det)-1)*np.power((mu-1)*(mu-1)*(1-1/tau_n), -1)

            f_vT = mu-1.
            c = (f_vT / 2) * erfc(-(f_vT) / np.sqrt(2 * sigma)) + np.power(2 * np.pi * sigma, -1 / 2) * sigma * np.exp(
                -f_vT * f_vT / (2 * sigma))

            lin_resp_full = (1/(2*c))*sigma*erfc(-(f_vT)/np.sqrt(2*sigma))*(1 - np.exp(-t_det)*np.exp(-t_det/tau_n))*np.power((mu-1)*(1+1/tau_n), -1)
            lin_resp_sim = (sum(eta)/len(eta))*(1 - np.exp(-t_det)*np.exp(-t_det/tau_n))*np.power((mu-1)*(1+1/tau_n), -1)

            lin_resp_full = np.mean(eta)*(1 - np.exp(-t_det)*np.exp(-t_det/tau_n))*np.power((mu-1)*(1+1/tau_n), -1)
            print(np.mean(eta))

            nonlin_resp = sigma*np.exp(-2*t_det)*np.power(mu-1., -2)*((np.exp(2*t_det) -1)/(2*(1+1/tau_n)) - (np.exp((1-1/tau_n)*t_det)-1)/((1+1/tau_n)*(1-1/tau_n)))
            nonlin_resp_full = (1/(2*c))*erfc(-(mu-1)/np.sqrt(2*sigma))*sigma*np.exp(-2*t_det)*np.power(mu-1., -1)*((np.exp(2*t_det) -1)/(2*(1+1/tau_n)) - (np.exp((1-1/tau_n)*t_det)-1)/((1+1/tau_n)*(1-1/tau_n)))

            data_chi1.append(-lin_resp_full/t_det)
            if(np.sqrt(np.var(t))/np.mean(t) > 0.4 and cv == False):
                print(np.sqrt(np.var(t)) / np.mean(t))
                cv = True
                cv_line = D
            #violin_parts = ax.violinplot(delta_t, [sigma], points=40, widths=0.1,
            #       showmeans=False, showextrema=False, showmedians=False, bw_method='silverman')
            #for pc in violin_parts["bodies"]:
            #    pc.set_color('C0')
            #    pc.set_facecolor('C0')
            #    pc.set_edgecolor('C0')
            #    pc.set_alpha(1)
        ax.scatter(Ds, plot_data, label = [r"$\tau_\eta=0.0$", r"$\tau_\eta=0.1$", r"$\tau_\eta=1.0$", r"$\tau_\eta=3.0$"][i], c=["k", "C0", "C1", "C2"][i])
        ax.plot(Ds, data_chi1, c=["k", "C0", "C1", "C2"][i])
    if(cv):
        ax.axvline(cv_line, c="C7", ls="--")
        ax.axvspan(cv_line, max(ax.get_xlim()), facecolor="C7", alpha=0.5)
    ax.tick_params(direction='in')
    plt.legend()

    plt.tight_layout()
    plt.savefig(home + "/Data/LIF/red/plots/real_eta_full_delta_t_sigma_mu{:.2f}.pdf".format(mu))
    plt.show()

    return 1

if __name__ == "__main__":
    plt_mean_ISI_der_sigma()