import matplotlib.pyplot as plt
from utilites import plot_parameters as utl
from utilites import functions as fc
import numpy as np
import matplotlib.patches as mpatches
import os

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
    print(utl.adjust_plotsize(1., 0.5))
    f = plt.figure(1, figsize=utl.adjust_plotsize(1., 0.5))
    x0 = 0.05
    x1 = 0.05
    y0 = 0.2
    y1 = 0.10
    hspacer = 0.05
    wspacer = 0.10
    height = (1 - hspacer - y0 - y1)
    width = (1 - 2*wspacer - x0 - x1) /3
    ax_1 = f.add_axes([x0, y0, width, height])
    ax_2 = f.add_axes([x0 + wspacer + width, y0, width, height], sharey=ax_1)
    ax_3 = f.add_axes([x0 + 2*wspacer + 2*width, y0, width, height], sharey = ax_1)
    axis = [ax_1, ax_2, ax_3]
    for ax in axis:
        #ax.grid(which='major', alpha=0.8, linestyle="--")
        ax.tick_params(direction='out', labelsize=utl.labelsize)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        # Only show ticks on the left and bottom spines
        ax.spines['left'].set_visible(False)
        ax.set_axisbelow(False)
        ax_3.set_yticks([])
        ax_3.xaxis.set_ticks_position('bottom')


    home = os.path.expanduser("~")
    data_full = home + "/Data/QIF/data_full/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(mu, tau_a, tau_n, D, delta, 0)
    vt, at, a_det, t_det = calculate_qif_timeseries(mu, tau_a, delta)
    prc = qif_prc(t_det, a_det, mu, tau_a)
    max_prc = max(prc)
    prc = [x/max_prc for x in prc]
    data = np.loadtxt(data_full)
    ts, vs, a, etas  = np.transpose(data)
    index_ti = [i for i, v in enumerate(vs) if v==-10_000]
    ti = [t for t, v in zip(ts, vs) if v==-10_000]
    ai = [x for x, v in zip(a, vs) if v==-10_000]

    s0 = 8
    spikes = 1
    start = index_ti[s0]-200
    stop = index_ti[spikes+s0]+200
    t_plt = ts[start:stop]
    eta_plt = etas[start:stop]

    eta_shifted = [x-0.2 for x in eta_plt]
    eta_max = max(eta_shifted)
    eta_plt  = [x/eta_max for x in eta_shifted]


    ax_1.plot(t_plt, eta_plt, zorder=2, c="C3", lw=0.7)
    ax_1.set_xticks([min(t_plt),  max(t_plt)])
    ax_1.set_ylim(-0.7, 1.6) #eta is between -1, 1.
    ax_1.text((min(t_plt) + max(t_plt))/2, 1.3, r"$u(t_i + \tau)$", ha="center", fontsize=utl.fontsize, clip_on=False, zorder=3)
    ax_1.set_xticklabels(["$t_i$","$t_i + T^*$"], fontsize=utl.fontsize, zorder=4)
    ax_1.set_title("Perturbation", fontsize=utl.fontsize)
    ax_1.arrow(max(t_plt)+0.4, 0.5, 0.8, 0, clip_on=False, head_width=0.2, head_length=0.5, lw=1.0, fc="k")


    n = int(len(eta_plt)/len(prc))
    conv = []
    for i in range(len(prc)):
        conv.append(2*prc[i]*eta_plt[n*i])
    ax_2.plot(np.linspace(0, t_det, 100), prc,  lw=1, c="C0")
    ax_2.set_xticks([0,  t_det])
    ax_2.set_xticklabels(["$t_i$","$t_i + T^*$"], fontsize=utl.fontsize)
    ax_2.set_title("Neuron", fontsize=utl.fontsize)
    ax_2.text(t_det/2, 1.3, r"$Z(\tau)u(t_i + \tau)$", ha="center", fontsize=utl.fontsize, clip_on=False)
    #ax_2.set_xlim([0-t_det/10, t_det+t_det/10])
    ax_2.plot(np.linspace(0, t_det, 100), conv, lw=1, c="C3")
    ax_2.fill_between(np.linspace(0, t_det, 100), prc, 0, facecolor="C0", alpha=0.5, zorder=2)
    ellipse = mpatches.Ellipse((t_det/2 + 0.2, 0.2), t_det*1.5, 2, color='k', fc="C7", alpha = 0.5, clip_on=False)
    ax_2.add_artist(ellipse)
    ax_2.arrow(t_det + 0.5, 0.5, 1, 0, clip_on=False, head_width=0.2, head_length=0.5, lw =1.0, fc="k")

    ax_3.set_title("Spike timing", fontsize=utl.fontsize)
    ax_3.set_xticks([0, t_det - t_det/4, t_det])
    ax_3.set_xticklabels(["$t_i$", "$t_{i+1}$", "$\quad t_i + T^*$"], fontsize=utl.fontsize)
    ax_3.set_xlim(0 -t_det/10, t_det +t_det/10)
    ax_3.arrow(t_det, 0.7, -(t_det/4), 0, length_includes_head=True, head_width=0.1, head_length=0.5, lw =1.0, ec="C3", fc="C3")
    ax_3.text(t_det - t_det/8, 0.9, "$\delta T_{i+1}$", ha="center", fontsize=utl.fontsize, clip_on=False)
    ax_3.text(t_det/2, 1.3, r"$\int_0^{T^*} d\tau\, Z(\tau)u(t_i + \tau)$", ha="center", fontsize=utl.fontsize, clip_on=False, zorder=3)

    plt.plot((0,0), (0, 0.75),c="k")
    plt.plot((t_det,t_det), (0, 0.75), ls="--", c="k")
    plt.plot((t_det - t_det/4, t_det - t_det/4), (0, 0.75), c="k")
    plt.savefig(home + "/Data/Plots_paper/1_lin_resp.pdf", transparent = True)

    plt.show()
    return 1

if __name__ == "__main__":
    mu = 5
    tau_a = 6
    tau_n = 4.0
    delta = 3
    D = 2.
    illustration()