import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

import styles as st

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
    data_full = home + "/Data/QIF/time_series_mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}.txt".format(mu, tau_a, tau_n, D, delta, 0)

    # Get Data:
    vt, at, a_det, t_det = calculate_qif_timeseries(mu, tau_a, delta)

    # Set up figure
    st.set_default_plot_style()
    fig = plt.figure(tight_layout=True, figsize=(3.5, 2.5))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])

    axis = [ax1]
    st.remove_top_right_axis(axis)


    #axis_ts: Plot time series
    data = np.loadtxt(data_full)
    ts, vs, a, etas  = np.transpose(data)

    index_ti = [i for i, v in enumerate(vs) if v==-10_000]

    s0 = 6
    start = index_ti[s0]-200
    stop = index_ti[s0+3]
    v_plt_pp = vs[start+199:stop+200]
    a_plt_pp = a[start+199:stop+200]

    # axis_pp: Plot phase portrait
    ax1.plot([np.arctan(v) for v in v_plt_pp], a_plt_pp, lw=1, c=st.colors[1])
    ax1.plot([np.arctan(v) for v in vt[:-2]], at[:-2], lw=2, c="k")
    ax1.plot([np.arctan(v) for v in vt[-3:]], at[-3:], lw=1, ls=":", c="k")
    ax1.scatter([np.pi/2, -np.pi/2], [at[-3], at[-1]], marker="s", s=20, ec="k", fc="w", zorder=4)
    ax1.text(-np.pi/2, 1.15*at[-1], r"$\phi = 0$", fontsize=13, va="top", ha="left")
    ax1.text(np.pi/2, 0.65*at[-3], r"$\phi = 2\phi$", fontsize=13, va="bottom", ha="right")

    ax1.text(0, a_det*1.05, "reset", ha="center")
    ax1.text(np.pi/2*1.15, (a_det+min(at))*0.9/2, "jump", ha="center", rotation=-90)
    ax1.set_xlim([-np.pi/2*1.2, np.pi/2*1.4])
    ax1.set_xticks([-np.pi/2, 0 , np.pi/2])
    ax1.set_xticklabels([r"$-\pi$", "$0$",r"$\pi$"])
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylim([0.6*at[-3], 1.2*a_det])
    ax1.set_yticks([a_det])
    ax1.set_yticklabels(["$a^*$"])
    ax1.set_ylabel("$a$")

    plt.savefig(home + "/Desktop/Presentations/SCC SfB/fig2.png", dpi=300)
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