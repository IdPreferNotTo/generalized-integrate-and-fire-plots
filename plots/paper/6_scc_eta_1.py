import matplotlib.pyplot as plt
import numpy as np
from utilites import functions as fc
from utilites import plot_parameters as utl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import chain
import os


def chunks(lst, n):
    m = int(len(lst)/n)
    for i in range(0, len(lst), m):
        yield lst[i:i+m]

def get_gif_t_a_w_det(w0, gamma, mu, beta, tau_w, tau_a, delta):
    v = 0
    w = w0
    a = delta/tau_a
    t = 0
    t1 = 0
    t_det = 0
    w_det = 0
    dt = 10 ** (-5)
    a_tmp0 = -1
    a_tmp1 = 0
    while abs(a_tmp0 - a_tmp1) > 0.0001:
        if v < 1.:
            v_tmp = v
            v += (mu - gamma*v - beta * w - a) * dt
            w += ((v_tmp - w) / tau_w) * dt
            a += (-a / tau_a) * dt
            t += dt
        else:
            t_det = t
            w_det = w
            a_tmp1 = a_tmp0
            a_tmp0 = a
            v = 0
            w = w0
            a += delta/tau_a
            t = 0
    return t_det, w_det, a


def gif_prc_theory(t_det, a_det, w_det, gamma, mu, tau_w, beta, tau_a, delta):
    prc = []
    nu = gamma + 1/tau_w
    w = np.sqrt((beta+gamma)/tau_w - nu**2/4)
    print( ((1.-tau_w*gamma)/(2*tau_w*w)))
    for t in np.linspace(0, t_det, 100):
        varprc = np.exp((nu/2)*(t-t_det))*(np.cos(w*(t-t_det)) - ((1.-tau_w*gamma)/(2*tau_w*w))*np.sin(w*(t-t_det)))/(mu-gamma-beta*w_det - a_det + delta/tau_a)
        prc.append(varprc)
    return prc


def scc_eta_1(gammas, mus, betas, tauws, tauas, deltas, sigmas, Dw, w_resets):
    home = os.path.expanduser('~')
    print(utl.adjust_plotsize(1., 0.5))
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., 0.5))
    x0 = 0.12
    x1 = 0.05
    y0 = 0.2
    y1 = 0.10
    hspacer = 0.05
    wspacer = 0.10
    height = (1 - hspacer - y0 - y1)
    width = (1 - wspacer - x0 - x1) /2
    ax_l = f.add_axes([x0, y0, width, height])
    ax_r = f.add_axes([x0 + wspacer + width, y0, width, height])

    axins_scc_l = inset_axes(ax_l, width="100%", height="100%", bbox_to_anchor=(.65, .15, .35, .35), bbox_transform=ax_l.transAxes)
    axins_prc_l = inset_axes(ax_l, width="100%", height="100%", bbox_to_anchor=(.15, .57, .35, .35), bbox_transform=ax_l.transAxes)

    axins_scc_r = inset_axes(ax_r, width="100%", height="100%", bbox_to_anchor=(.65, .15, .35, .35),
                           bbox_transform=ax_r.transAxes)
    axins_prc_r = inset_axes(ax_r, width="100%", height="100%", bbox_to_anchor=(.15, .57, .35, .35),
                           bbox_transform=ax_r.transAxes)
    axis_l = [ax_l, axins_prc_l, axins_scc_l]
    axis_r = [ax_r, axins_prc_r, axins_scc_r]
    axis_main = [ax_l, ax_r]
    axis = [axis_l, axis_r]
    for ax in axis_main:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which="both", direction='in', labelsize=utl.labelsize)
        ax.set_xlim([0.01, 100])
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_ylim([-0.7, 1])
        ax.set_yticks([-0.5, 0, 0.5, 1])
        ax.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
        ax.set_xlabel(r"$\tau_\eta/T^*$", fontsize=utl.fontsize)
    ax_l.set_ylabel(r"$\rho_{1}$", fontsize=utl.fontsize)
    for ax in [axins_prc_l, axins_prc_r, axins_scc_l, axins_scc_r]:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(which="both", direction='in', labelsize=utl.labelsize-2)

    axins_scc_l.set_ylim([-0.02, 0.12])
    axins_scc_l.set_yticks([0., 0.1])
    axins_scc_r.set_ylim([-0.24, 0.04])
    axins_scc_r.set_yticks([-0.2, 0])
    for axins_scc in [axins_scc_r, axins_scc_l]:
        axins_scc.set_xticks([1,5])
        axins_scc.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
        axins_scc.set_title(r"$\rho_{k}$", fontsize=utl.fontsize-1, rotation = "horizontal")
        axins_scc.set_xlabel(r"$k$", fontsize=utl.fontsize-1)
    for axins_prc in [axins_prc_r, axins_prc_l]:
        axins_prc.set_title(r"$Z(\tau)$", fontsize=utl.fontsize - 1, rotation = "horizontal")
        axins_prc.set_xlabel(r"$\tau/T^*$", fontsize=utl.fontsize - 1)

    axins_prc_l.set_yticks([-1, 0, 1])
    axins_prc_r.set_yticks([-0.2, 0, 0.2])
    ax_l.text(-0.2, 1.1, "(a)", size=10, weight = 'heavy', transform=ax_l.transAxes)
    ax_r.text(-0.2, 1.1, "(b)", size=10, weight = 'heavy', transform=ax_r.transAxes)

    for gamma, mu, beta, tauw, taua, delta, sigma, w_reset, axs in zip(gammas, mus, betas, tauws, tauas, deltas, sigmas, w_resets, axis):
        p1s_sim = []
        p1s_theory = []
        ax = axs[0]
        axins_prc = axs[1]
        axins_scc = axs[2]
        t_det, w_det, a_det = get_gif_t_a_w_det(w_reset, gamma, mu, beta, tauw, taua, delta)
        print(t_det, w_det, a_det)
        tau_ns = [0.01*t_det*(1.5**k) for k in range(25)]
        tau_n_t_det = []
        for tau_n in tau_ns:
            print("tau_n:", tau_n)
            Dn = sigma * tau_n
            home = os.path.expanduser('~')
            data_file = home + "/Data/GIF/data/mu{:.2f}_beta{:.1f}_tauw{:.1f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
                mu, beta, tauw, taua, delta, tau_n, Dn, Dw)
            print(data_file)
            data = np.loadtxt(data_file)
            t, a, eta, chi = np.transpose(data)
            t_det = fc.read_t_det(data_file)
            delta_t = [x - t_det for x in t]
            c0 = fc.k_corr(delta_t, delta_t, 0)
            c1 = fc.k_corr(delta_t, delta_t, 1)

            p1s_sim.append(c1/c0)
            p1s_theory.append(fc.GIF_scc(t_det, w_det, gamma, mu, tauw, beta, taua, delta, tau_n, Dn, Dw, 1))
            tau_n_t_det.append(tau_n/t_det)

        if w_reset == 0:
            tau_n_inset = 0.01*t_det*(1.5**2)
            ax.annotate("",
                        xy=(tau_n_inset/t_det, 0.0), xycoords='data',
                        xytext=(0.01 * t_det * (1.5 ** 8)/t_det*2, -0.45), textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="angle,angleA=180,angleB=-90,rad=0"),
                        )
        else:
            tau_n_inset = 0.01 * t_det * (1.5 ** 8)
            ax.annotate("",
                        xy=(tau_n_inset/t_det, -0.25), xycoords='data',
                        xytext=(tau_n_inset/t_det*2., -0.45), textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="angle,angleA=180,angleB=-90,rad=0"),
                        )
        Dn = sigma * tau_n_inset
        scc_theory = []
        scc_sim = []
        data_file = home + "/Data/GIF/data/mu{:.2f}_beta{:.1f}_tauw{:.1f}_taua{:.1f}_Delta{:.1f}_taun{:.3f}_Dn{:.2e}_Dw{:.2e}.txt".format(
            mu, beta, tauw, taua, delta, tau_n_inset, Dn, Dw)
        data = np.loadtxt(data_file)
        t, a, eta, chi = np.transpose(data)
        t_mean = np.mean(t)
        print("t_mean/t_det: ", t_mean / t_det)
        delta_t = [x - t_det for x in t]
        c0 = fc.k_corr(delta_t, delta_t, 0)
        for k in [1,2,3,4,5]:
            scc_theory.append(fc.GIF_scc(t_det, w_det, gamma, mu, tauw, beta, taua, delta, tau_n_inset, Dn, Dw, k))
            ck = fc.k_corr(delta_t, delta_t, k)
            scc_sim.append(ck/c0)

        prc = gif_prc_theory(t_det, a_det, w_det, gamma, mu, tauw, beta, taua, delta)
        axins_prc.plot(np.linspace(0, t_det, 100), prc, lw=1, c="k")
        axins_prc.set_xlim([0, t_det])
        prc_neg = [x for x in prc if x < 0]
        prc_pos = [x for x in prc if x >= 0]
        t0_step = len(prc_neg)
        t0 = t0_step / 100 * t_det
        axins_prc.set_xticks([0, t_det])
        axins_prc.set_xticklabels([0, 1])
        #axins_prc.set_yticks([0, 0.4])
        axins_prc.fill_between(np.linspace(0, t0, t0_step), prc_neg, 0, facecolor="C3", alpha=0.5, zorder=2)
        axins_prc.fill_between(np.linspace(t0, t_det, 100 - t0_step), prc_pos, 0, facecolor="C0", alpha=0.5, zorder=2)

        ax.axhline(0, xmin=0, xmax=6, ls="--", c="C7", zorder=1)
        ax.plot(tau_n_t_det, p1s_theory, c="k", lw=1, zorder=1)
        ax.scatter(tau_n_t_det, p1s_sim, fc="w", edgecolors="k", s=10, zorder=2)
        axins_scc.scatter([1,2,3,4,5], scc_sim, fc="w", edgecolors="k", s=10, zorder=2)
        axins_scc.plot([1,2,3,4,5], scc_theory, c="k", lw=1, zorder=1)
        #axins_scc.patch.set_alpha(1.0)
        #axins_prc.patch.set_alpha(1.0)

    plt.savefig(home + "/Data/Plots_paper/6_gif_p1_negative_prc.pdf".format(Dn))
    plt.show()

    return 1

if __name__ == "__main__":
    gammas = [-1, -1]
    mus = [1, 1]
    betas = [5, 5]
    tauws = [1.1, 1.1]
    tauas = [1, 1]
    deltas = [2.3, 0]
    sigmas = [0.001, 0.001]
    w_resets = [0, 1]
    Dw =  0
    scc_eta_1(gammas, mus, betas, tauws, tauas, deltas, sigmas, Dw, w_resets)