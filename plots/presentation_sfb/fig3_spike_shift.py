import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import os

from utilites import plot_parameters as utl
import styles as st

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


def illustration():
    vt, at, a_det, t_det = calculate_qif_timeseries(mu, tau_a, delta)

    st.set_default_plot_style()
    st.set_default_plot_style()
    fig = plt.figure(tight_layout=True, figsize=(6, 2))
    gs = gridspec.GridSpec(1, 3, wspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axis = [ax1, ax2, ax3]
    for ax in axis:
        ax.set_yticks([])
    st.remove_top_right_axis(axis)

    ax1.text(0.5, 0.85, r"$u(t_i + \tau)$", size=12, transform=ax1.transAxes, va="center", ha="center")
    ax1.text(0.5, 1.00, "Voltage pert.", size=13, transform=ax1.transAxes, va="center", ha="center")

    ax2.text(0.5, 0.85, r"$Z(\tau)u(t_i + \tau)$", size=12, transform=ax2.transAxes, va="center", ha="center")
    ax2.text(0.5, 1.00, "Phase pert.", size=13, transform=ax2.transAxes, va="center", ha="center")

    ax3.text(0.5, 0.85, r"$-\int_0^{T^*} d\tau\, Z(\tau)u(t_i + \tau)$", size=12, transform=ax3.transAxes, va="center", ha="center")
    ax3.text(0.5, 1.00, "Spike timing", size=13, transform=ax3.transAxes, va="center", ha="center")
    ax3.text(0.73, 0.45, "$\delta T_{i+1}$", size=12, transform=ax3.transAxes, va="center", ha="center")


    home = os.path.expanduser("~")
    data_full = home + "/Data/QIF/time_series_mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}.txt".format(mu, tau_a, tau_n, D, delta, 0)

    prc = qif_prc(t_det, a_det, mu, tau_a)
    max_prc = max(prc)
    prc = [x/max_prc for x in prc]
    data = np.loadtxt(data_full)
    ts, vs, a, etas  = np.transpose(data)
    index_ti = [i for i, v in enumerate(vs) if v==-10_000]

    s0 = 8
    spikes = 1
    start = index_ti[s0]-200
    stop = index_ti[spikes+s0]+200
    t_plt = ts[start:stop]
    eta_plt = etas[start:stop]

    eta_shifted = [x-0.2 for x in eta_plt]
    eta_max = max(eta_shifted)
    eta_plt  = [x/eta_max for x in eta_shifted]


    ax1.plot(t_plt, eta_plt, zorder=2, c=st.colors[1], lw=1.5)
    ax1.set_xticks([min(t_plt),  max(t_plt)])
    ax1.set_ylim(-0.7, 1.6) #eta is between -1, 1.
    ax1.set_xticklabels(["$t_i$","$t_i + T^*$"], fontsize=utl.fontsize, zorder=4)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim([1.3*ymin, 1.3*ymax])
    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim([xmin-t_det/10, xmax+t_det/10])
    #ax1.arrow(max(t_plt)+0.4, 0.5, 0.8, 0, clip_on=False, head_width=0.2, head_length=0.5, lw=1.0, fc="k")


    n = int(len(eta_plt)/len(prc))
    conv = []
    for i in range(len(prc)):
        conv.append(2*prc[i]*eta_plt[n*i])
    ax2.plot(np.linspace(0, t_det, 100), prc,  lw=1, c="k")
    ax2.set_xticks([0,  t_det])
    ax2.set_xticklabels(["$t_i$","$t_i + T^*$"], fontsize=utl.fontsize)
    #ax_2.set_xlim([0-t_det/10, t_det+t_det/10])
    ax2.plot(np.linspace(0, t_det, 100), conv, lw=1.5, c=st.colors[1])
    ax2.fill_between(np.linspace(0, t_det, 100), prc, 0, facecolor= st.colors[1], alpha=0.4, zorder=2)
    #ax2.arrow(t_det + 0.5, 0.5, 1, 0, clip_on=False, head_width=0.2, head_length=0.5, lw =1.0, fc="k")
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim([ymin, 1.5*ymax])
    xmin, xmax = ax2.get_xlim()
    ax2.set_xlim([xmin-t_det/10, xmax+t_det/10])

    ax3.set_xticks([0, t_det - t_det/3, t_det])
    ax3.set_xticklabels(["$t_i$", "$t_{i+1}$", "$\quad t_i + T^*$"], fontsize=utl.fontsize)
    ax3.set_xlim(0 -t_det/10, t_det +t_det/10)
    ax3.arrow(t_det, 0.7, -(t_det/3), 0, length_includes_head=True, head_width=0.1, head_length=0.5, lw =1.0, ec=st.colors[1], fc=st.colors[1])
    ax3.plot((0,0), (0, 0.75),c="k")
    ax3.plot((t_det,t_det), (0, 0.75), ls="--", c="k")
    ax3.plot((t_det - t_det/3, t_det - t_det/3), (0, 0.75), c="k")
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim([ymin, 1.5*ymax])
    xmin, xmax = ax3.get_xlim()
    ax3.set_xlim([xmin-t_det/10, xmax+t_det/10])

    plt.savefig(home + "/Desktop/Presentations/SCC SfB/fig3.png", dpi=300)
    plt.show()
    return 1


if __name__ == "__main__":
    mu = 5
    tau_a = 6
    tau_n = 4.0
    delta = 3
    D = 2.
    illustration()