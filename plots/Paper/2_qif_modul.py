import numpy as np
import matplotlib.pyplot as plt
import os
import cmath

import styles
from utilites import plot_parameters as utl
from utilites import functions as fc
import styles as st

def adap_cnoise_QIF_scc(t_det, prc, tau_a, tau_n, delta, k):
    alpha = np.exp(-t_det / tau_a)
    beta = np.exp(-t_det/tau_n) if tau_n != 0 else 0
    a_det = delta / (1 - alpha)

    zi = 0
    dt = t_det/len(prc)
    for n in range(len(prc)):
        zi += prc[n]*np.exp(-n*dt/tau_a)*dt
    nu = 1. - (a_det/tau_a)*zi

    if tau_n==0:
       chi2 = 0
       chi1 = 1
    else:
        chi2 = 0
        chi1 = 0
        dt = t_det / 100
        for n in range(100):
            for m in range(100):
                chi2 += prc[n] * prc[m] * np.exp(-(n - m)*dt / tau_n) * dt * dt
                chi1 += prc[n] * prc[m] * np.exp(-abs(n - m)*dt / tau_n) * dt * dt

    c1 = (1. + (alpha * nu) ** 2 - 2*alpha * nu * beta) / (alpha * nu - beta)
    c2 = (1. - (alpha * nu) ** 2) * (1 - alpha * beta) * (alpha - beta) / (
            (alpha * nu - beta) * (1. + alpha ** 2 - 2 * nu * alpha ** 2))
    c3 = 1. - alpha * nu * beta
    A = lambda i: alpha*(1 - alpha*alpha*nu)*(1-nu)*np.real(cmath.exp((i-1) * cmath.log(alpha*nu)))
    C = 1+ alpha**2 - 2 * alpha**2 * nu

    pa = lambda i : -A(i)/C
    pn = lambda i: beta**(i)*chi2/chi1

    factor1 = ((c1*pn(1) +c3)/(2*pa(1)*pn(1) + c3))
    factor2 = (c2/(2*pa(1)*pn(1) + c3))
    print(2*pa(1)*pn(1) + c3)
    scc = []
    for n in range(1, k+1):
        scc.append(factor1*pa(n) + factor2*pn(n))
    return scc


def qif_prc(t_det, a_det, mu, tau_a):
    dt = 10**(-5)
    epsilon = 0.01
    ts=np.linspace(0, t_det, 100)
    prc = []
    for tk in ts:
        kick=True
        v = -10_000
        a = a_det
        t = 0
        while v < 10_000:
            if (t>=tk and kick==True):
                v = v+epsilon
                kick=False
            v += (mu + v**2 - a)*dt
            a += (-a/tau_a)*dt
            t += dt
        prc.append(-(t - t_det)/epsilon)
    return prc


def calculate_qif_timeseries(beta, tau_a, delta):
    vt = []
    at = []

    v = 0
    a = 0
    t = 0

    dt = 10**(-5)
    a_tmp0 = -1
    a_tmp1 = 0

    osc = 0
    count = 0
    while abs(a_tmp0 - a_tmp1) > 0.001:
        if v < 10_000:
            v += (beta + v**2  - a)*dt
            a += (-a/tau_a)*dt
        else:
            v = -10_000
            a += delta
            a_tmp1 = a_tmp0
            a_tmp0 = a
    while v < 10_000:
        if count%100==0:
            vt.append(v)
            at.append(a)
        v += (beta + v ** 2 - a) * dt
        a += (-a / tau_a) * dt
        t += dt
        count += 1
    vt.append(10_000)
    at.append(a)
    a += delta
    vt.append(10_000)
    at.append(a)
    v = -10_000
    vt.append(v)
    at.append(a)
    osc += 1
    return vt, at, a, t


def plt_qif_model(tau_a, tau_n, mu, delta, D):
    home = os.path.expanduser('~')
    data_file = home + "/Data/integrate_and_fire/quadratic_if/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}.txt".format(mu, tau_a, tau_n, D, delta, 0)
    data_full = home + "/Data/integrate_and_fire/quadratic_if/time_series_mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}.txt".format(mu, tau_a, tau_n, D, delta, 0)
    print(utl.adjust_plotsize(1.))
    # Get Data:
    vt, at, a_det, t_det = calculate_qif_timeseries(mu, tau_a, delta)
    prc = qif_prc(t_det, a_det, mu, tau_a)
    print(t_det)
    print(utl.adjust_plotsize(1.))
    # Set up figure
    st.set_default_plot_style()
    colors = st.Colors()
    f = plt.figure(1, figsize=utl.adjust_plotsize(1.))
    x0 = 0.11
    y0 = 0.12
    spacer = 0.025
    width = 0.5 - x0 - spacer
    height = 0.33 - y0
    hspacer = 0.01
    ax_vt = f.add_axes([x0, y0 + 0.33 + 4/3*(height-hspacer)+2*hspacer, width, 2/3*(height-hspacer)])
    ax_at = f.add_axes([x0, y0 + 0.33 + 2/3*(height-hspacer)+hspacer, width, 2/3*(height-hspacer)])
    ax_nt = f.add_axes([x0, y0 + 0.33, width, 2/3*(height-hspacer)])
    ax_pp = f.add_axes([x0 + 0.5, y0 + 0.33, width, 2*height])
    ax_prc = f.add_axes([x0, y0, width, height])
    ax_scc = f.add_axes([x0 + 0.5, y0, width, height])
    axis_ts = [ax_vt, ax_at, ax_nt]
    axis = [ax_vt, ax_at, ax_nt, ax_pp, ax_prc, ax_scc]

    for ax in axis:
        ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(direction='in', labelsize=utl.labelsize)

    # set axis labels
    ax_vt.text(-1.6, 1.1, "A", size=12, transform=ax_pp.transAxes)
    ax_pp.text(-0.2, 1.1, "B", size=12, transform=ax_pp.transAxes)
    ax_prc.text(-1.6, 1.1, "C", size=12, transform=ax_scc.transAxes)
    ax_scc.text(-0.2, 1.1, "D", size=12, transform=ax_scc.transAxes)

    # ax_ul: Plot phase response curve
    ax_prc.plot(np.linspace(0, t_det, 100), prc, lw=1, c="k")
    ax_prc.set_xticks([0, t_det/2, t_det])
    ax_prc.set_xticklabels([0, 1/2, 1], fontsize=11)
    ax_prc.set_xlabel(r"$\tau/T^*$", fontsize=11)
    ax_prc.set_ylabel(r"$Z(\tau)$", fontsize=11)
    ax_prc.set_xlim([0, t_det])
    ax_prc.set_ylim(0, max(prc)*1.2)
    ax_prc.fill_between(np.linspace(0, t_det, 100), prc, 0, facecolor=colors.palette[1], alpha=0.5, zorder=2)

    # ax_scc: Plot serial correlation coefficients.
    # For adaptation, colored noise and a combination of both
    print(data_file)
    data = np.loadtxt(data_file)
    t, delta_a, eta, chi, chi2 = np.transpose(data)
    t_mean = np.mean(t)
    print(t_mean)
    delta_t = [x - t_mean for x in t]
    corr_sim = []

    variance_delta_t = fc.k_corr(delta_t, delta_t, 0)
    k_range = range(1, 7)
    c0 =fc.k_corr(delta_t, delta_t, 0)
    cv = np.sqrt(c0) / t_mean
    print(cv)
    print("k: ", 0)
    print("<tt>: ", variance_delta_t)
    print("<ta>: ", fc.k_corr(delta_t, delta_a, 0))
    print("<tn>: ", -fc.k_corr(delta_t, eta, 0))
    print("k: ", 1)
    print("<tt>: ", variance_delta_t)
    print("<ta>: ", fc.k_corr(delta_t, delta_a, 1))
    print("<tn>: ", -fc.k_corr(delta_t, eta, 1))
    for k in k_range:
        covariance_delta_t = fc.k_corr(delta_t, delta_t, k)
        corr_sim.append(covariance_delta_t / variance_delta_t)

    corr_theory = adap_cnoise_QIF_scc(t_det, prc, tau_a, tau_n, delta, max(k_range))
    ax_scc.scatter(k_range, corr_sim, ec="k", fc="w", label="sim.", zorder=2)
    ax_scc.plot(k_range, corr_theory, c="k", ls="-", lw=1, label="theory", zorder=1)
    ax_scc.set_xticks(k_range)
    ax_scc.set_ylabel(r"$\rho_k$")
    #ax_scc.set_ylim([-0.1, 0.05])
    ax_scc.set_xlabel("$k$")
    ax_scc.set_ylim([-0.1, 0.1])
    ax_scc.legend(prop={"size":7}, loc=1, ncol=2, framealpha=1., edgecolor="k")
    leg = ax_scc.get_legend()
    leg.get_frame().set_linewidth(0.5)

    #axis_ts: Plot time series
    data = np.loadtxt(data_full)
    ts, vs, a, etas  = np.transpose(data)

    index_ti = [i for i, v in enumerate(vs) if v==-10_000]
    ti = [t for t, v in zip(ts, vs) if v==-10_000]
    ai = [x for x, v in zip(a, vs) if v==-10_000]
    ISI = [i-j-t_mean for (i, j) in zip(ti[1:], ti[:-1])]
    print(ISI)

    s0 = 6
    spikes = 3
    start = index_ti[s0]-200
    stop1 = index_ti[s0+1]
    stop = index_ti[spikes+s0]+200
    t_plt = ts[start:stop]
    t_min = ts[start]
    t_max = ts[stop]
    v_plt = vs[start:stop]
    v_plt_pp = vs[start+199:stop1+200]
    a_plt = a[start:stop]
    a_plt_pp = a[start+199:stop1+200]
    eta_plt = etas[start:stop]

    for ax in axis_ts:
        ax.set_xlim(t_min, t_max)
        ax.set_xticks(ti[s0:spikes+s0+1])
        xlabels = []
        xlabels.append("$t_{i-1}$")
        xlabels.append("$t_i$")
        for s in range(s0, s0+spikes)[1:]:
            xlabels.append("$t_{{i+{:d}}}$".format(s-s0))
    for s in range(s0, s0+spikes):
        if s  == s0:
            ax_vt.text((ti[s] + ti[s+1]) / 2, 2.9, "$T_{{i}}$", clip_on = False, ha="center", fontsize=11)
        else:
            ax_vt.text((ti[s] + ti[s+1])/2, 2.9, "$T_{{i+{:d}}}$".format(s-s0), ha="center", clip_on = False, fontsize=11)
        #axis[0].plot([ti[s], ti[s+1]], [1.5, 1.5], c = "k")
        ax_vt.arrow(ti[s], 2.5 , ti[s+1] - ti[s], 0, length_includes_head=True, head_width=0.5, head_length=0.0, lw =0.5, clip_on = False)
    ax_vt.arrow(ti[s0], 2.5, 0.001 , 0, length_includes_head=True, head_width=0.5, head_length=0.0, lw=0.5, clip_on = False)

    ax_vt.set_ylim([-np.pi/2-0.4, np.pi/2 + 0.4])
    ax_vt.set_yticks([-np.pi/2, np.pi/2])
    ax_vt.set_yticklabels([r"$\theta_R$", r"$\theta_T$"])
    ax_vt.set_xticklabels([])
    ax_vt.plot(t_plt, [np.arctan(v) for v in v_plt], zorder=2, lw = 1, c=colors.palette[1])
    ax_vt.set_ylabel(r"$\theta$", fontsize=11)

    ax_at.plot(t_plt, a_plt, zorder=2, c=colors.palette[3], lw = 1)
    ax_at.set_ylim([min(a_plt)*0.8, a_det*1.2])
    ax_at.set_yticks([a_det])
    ax_at.set_yticklabels(["$a^*$"])
    ax_at.set_xticklabels([])
    ax_at.set_ylabel("$a$", fontsize=11)
    ax_at.text(ti[s0 + 1], a_det, r"$a_i$", ha="center", fontsize=11)
    ax_at.text(ti[s0 + 2], a_det, r"$a_{i+1}$", ha="center", fontsize=11)

    ax_nt.plot(t_plt, eta_plt, zorder=2, c=colors.palette[5], lw = 0.5)
    ax_nt.set_yticks([0])
    ax_nt.set_ylabel(r"$\eta$", fontsize=11)
    ax_nt.set_xticklabels(xlabels)
    ax_nt.set_xlabel("$t$")
    ymax = abs(min(eta_plt)) if abs(min(eta_plt)) > abs(max(eta_plt)) else abs(max(eta_plt))
    ax_nt.set_ylim([-ymax, ymax])

    # axis_pp: Plot phase portrait
    ax_pp.plot([np.arctan(v) for v in v_plt_pp], a_plt_pp, lw=1, c=colors.palette[1])
    ax_pp.plot([np.arctan(v) for v in vt], at, ls="--", lw=1, c="k")
    ax_pp.text(0, a_det*1.05, "reset", ha="center", fontsize=11)
    ax_pp.text(np.pi/2*1.15, (a_det+min(at))*0.9/2, "jump", ha="center", fontsize=11, rotation=-90)
    ax_pp.set_xlim([-np.pi/2*1.2, np.pi/2*1.4])
    ax_pp.set_xticks([-np.pi/2, 0 , np.pi/2])
    ax_pp.set_xticklabels([r"$\theta_R$", "$0$",r"$\theta_T$"])
    ax_pp.set_xlabel(r"$\theta$", fontsize=11)
    da = 0.2*a_det
    ax_pp.set_ylim([min(a_plt_pp)-da, a_det + da])
    ax_pp.set_yticks([a_det])
    ax_pp.set_yticklabels(["$a^*$"])
    ax_pp.set_ylabel("$a$", fontsize=11)

    plt.savefig(home + "/Desktop/bccn_conference_plots/fig2.png", dpi=300)
    plt.show()

    return 1

if __name__ == "__main__":
    for m in range(1):
        for k in range(1):
            for i in range(1):
                mu = 5
                tau_a = [6, 10][m]
                tau_n = [4.0][k]
                delta = [3, 1][m]
                D = 2.
                plt_qif_model(tau_a, tau_n, mu, delta, D)