import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
from utilites import plot_parameters as utl
from matplotlib.collections import LineCollection


def k_corr(data1, data2, k):
    # Get two arbitrary data set and calculate their correlation with lag k.
    mean_d1 = np.mean(data1)
    mean_d2 = np.mean(data2)
    data1 = [x - mean_d1 for x in data1]
    data2 = [x - mean_d2 for x in data2]

    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])


if __name__ == "__main__":
    home = os.path.expanduser("~")
    data = np.loadtxt(home + "/Data/Taub_miles/PRC_I5.0.dat")
    taus, PRCs = np.transpose(data)

    # I = 10.
    ISI = 10.1715
    tau_a = 1 / 0.01
    z_fix = 0.0452
    g = 5
    deltaV = 33
    a_fix = g * deltaV * z_fix
    Delta = tau_a * a_fix * (1. - np.exp(-ISI / tau_a))
    print(Delta)

    # I = 5.
    ISI = 18.98
    tau_a = 100
    z_fix = 0.0254
    g = 5
    deltaV = 33
    a_fix = g * deltaV * z_fix
    Delta = tau_a * a_fix * (1. - np.exp(-ISI / tau_a))

    Dw = 0.1
    Dn = 1
    tau_n = 10

    fig = plt.figure(tight_layout=True, figsize=utl.adjust_plotsize(1., 0.5))
    gs = gs.GridSpec(2, 2)
    ax11 = fig.add_subplot(gs[0,0:1])
    ax12 = fig.add_subplot(gs[1,0:1])
    ax2 = fig.add_subplot(gs[:, 1:2])

    ax11.text(-0.2, 1.2, "(a)", size=10, weight = 'heavy', transform=ax11.transAxes)
    ax2.text(-0.2, 1.1, "(b)", size=10, weight = 'heavy', transform=ax2.transAxes)

    ax11.grid(which='major', alpha=0.8, linestyle="--")
    ax11.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax11.set_ylabel(r"$Z(\tau)$", fontsize=utl.fontsize)
    ax11.set_xticks([0, 0.5*ISI, ISI])
    ax11.set_xticklabels([])
    ax11.plot(taus, PRCs, c="k")
    ax11.fill_between(taus, PRCs, 0, facecolor="C0", alpha=0.5, zorder=2)
    ax11.set_xlim([0, ISI])
    #ax11.set_ylim([0, 0.5])

    data = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/PRC/I5.0/taubs_miles_adap_cnoiseI5.0_t20.00_e0.10.dat")
    v, a, eta, n, m , h, Ia, t = np.transpose(data)

    ax12.plot(t, v, c="k")
    ax12.grid(which='major', alpha=0.8, linestyle="--")
    ax12.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax12.set_xlabel(r"$\tau / T^*$ ", fontsize=utl.fontsize)
    ax12.set_ylabel(r"$V$ [mV]", fontsize=utl.fontsize)
    ax12.set_xlim([0, ISI])
    ax12.set_xticks([0, 0.5 * ISI, ISI])
    ax12.set_xticklabels([0, 0.5, 1])
    ax12.set_ylim([-100, 60])
    ax12.set_yticks([-100, -50, 0, 50])


    ax2.grid(which='major', alpha=0.8, linestyle="--")
    ax2.set_ylabel(r"$\rho_k$", fontsize=utl.fontsize)
    ax2.set_xlabel("$k$", fontsize=utl.fontsize)
    ax2.tick_params(which="both", direction='in', labelsize=utl.labelsize)
    ax2.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
    ISIs_adap = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/spikes_taubs_miles_adap_cnoise_I5.0_Dv0.10_Dn0.00.dat")
    ISIs_adap = ISIs_adap[100:]

    ISIs_cnoise = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/spikes_taubs_miles_cnoise_I1.4_Dv0.10_Dn1.00.dat")
    ISIs_cnoise = ISIs_cnoise[100:]

    ISIs_adap_cnoise = np.loadtxt(home + "/CLionProjects/PhD/taub_miles_adap/out/spikes_taubs_miles_adap_cnoise_I5.0_Dv0.10_Dn1.00.dat")
    ISIs_adap_cnoise = ISIs_adap_cnoise[100:]

    ks = np.arange(1, 8)

    k_correlatins_adap = []
    std = k_corr(ISIs_adap, ISIs_adap, 0)
    for k in ks:
        corr = k_corr(ISIs_adap, ISIs_adap, k)
        k_correlatins_adap.append(corr / std)

    k_correlatins_cnoise = []
    std = k_corr(ISIs_cnoise, ISIs_cnoise, 0)
    for k in ks:
        corr = k_corr(ISIs_cnoise, ISIs_cnoise, k)
        k_correlatins_cnoise.append(corr/std)

    k_correlatins_adap_cnoise = []
    std = k_corr(ISIs_adap_cnoise, ISIs_adap_cnoise, 0)
    print(std)
    for k in ks:
        corr = k_corr(ISIs_adap_cnoise, ISIs_adap_cnoise, k)
        k_correlatins_adap_cnoise.append(corr / std)

    data = np.loadtxt(home + "/Data/Taub_miles/PRC_I5.0.dat")

    alpha = np.exp(-ISI/tau_a)
    beta  = np.exp(-ISI/tau_n)

    rhos_a = []
    rhos_n = []
    rhos = []

    ts, prc = np.transpose(data)
    dt = ts[1] - ts[0]
    integral = 0
    for t, Z in zip(ts, prc):
        integral += Z*np.exp(-t/tau_a)*dt
    nu = 1. - a_fix/tau_a * integral
    rho_1_a = - (1.-nu)*alpha*(1. - (alpha**2)*nu)/(1 + alpha**2 -2*(alpha**2)*nu)
    rhos_a = []
    for k in ks:
        rho_k_a = rho_1_a*(alpha*nu)**(k-1)
        rhos_a.append(rho_k_a)

    chi2 = 0
    chi1 = 0
    for t1, Z1 in zip(ts, prc):
        for t2, Z2 in zip(ts, prc):
            chi2 += Z1 * Z2 * np.exp(-abs(ISI + t2 - t1) / tau_n) * dt * dt
            chi1 += Z1 * Z2 * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
    for t1, Z1 in zip(ts, prc):
        chi1 += (2*tau_n * Dw / Dn) * Z1 ** 2 * dt
    for k in ks:
        rho_k_n = beta**(k-1) * chi2 / chi1
        rhos_n.append(rho_k_n)

    A = 1 + (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta) / ((alpha * nu - beta)) * rhos_n[0] - alpha * nu * beta
    B = (1. - (alpha * nu) ** 2) * (1. - alpha * beta) * (alpha - beta) / ((1 + alpha ** 2 - 2 * alpha ** 2 * nu) * (alpha * nu - beta))
    C = 1 + 2 * rhos_a[0] * rhos_n[0] - alpha * nu * beta
    for rho_a, rho_n in zip(rhos_a, rhos_n):
        rhos.append(A/C * rho_a + B/C * rho_n)


    ax2.scatter(ks, k_correlatins_adap, ec="k", s=20, fc = "C0", linewidths=1, zorder=5)
    ax2.scatter(ks, k_correlatins_cnoise, ec="k", s=20, fc = "C1", linewidths=1, zorder=5)
    ax2.scatter(ks, k_correlatins_adap_cnoise, ec="k", s=20, fc = "w", linewidths=1, zorder=5, label="sim.")
    ax2.plot(ks, rhos_a, c="C0")
    ax2.plot(ks, rhos_n, c="C1")
    ax2.plot(ks, rhos, c="k", label="theory")
    ax2.text(2.5, -0.20, r"$\rho_{k,a}$", ha="center", fontsize=utl.fontsize, c="C0")
    ax2.text(3.0, 0.10, r"$\rho_{k,\eta}$", ha="center", fontsize=utl.fontsize, c="C1")

    ax2.set_xticks([2, 4, 6])
    ax2.set_ylim([-0.3, 0.3])
    ax2.legend(prop={"size": 7}, framealpha=1., edgecolor="k")
    plt.savefig(home + "/Data/Taub_miles/Plots/traub_miles_prc_and_scc_2.pdf")
    plt.show()


