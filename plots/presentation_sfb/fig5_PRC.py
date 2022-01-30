import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import styles as st
from utilites import functions as fc

def lif_prc_theory(t_det, a_det, mu, tau_a, delta):
    prc = []
    for t in np.linspace(0, t_det, 100):
        varprc = np.exp(t - t_det)/(mu-1.-a_det+delta/tau_a)
        prc.append(varprc)
    return prc

def gif_prc_theory(t_det, a_det, w_det, gamma, mu, tau_w, beta, tau_a, delta):
    prc = []
    nu = gamma + 1/tau_w
    w = np.sqrt((beta+gamma)/tau_w - nu**2/4)
    for t in np.linspace(0, t_det, 100):
        varprc = np.exp((nu/2)*(t-t_det))*(np.cos(w*(t-t_det)) - ((1.-tau_w*gamma)/(2*tau_w*w))*np.sin(w*(t-t_det)))/(mu-gamma-beta*w_det - a_det + delta/tau_a)
        prc.append(varprc)
    return prc

if __name__ == "__main__":
    mu1 = 5
    taua1 = 2
    delta1 = 2
    t_det1, a_det = fc.get_lif_t_a_det(mu1, taua1, delta1)
    prc_typ1 = lif_prc_theory(t_det1, a_det, mu1, taua1, delta1)

    gamma2 = -1
    mu2 = 1
    taua2 = 1
    delta2 = 2.3
    beta2 = 5
    tauw2 = 1.1
    t_det2, w_det, a_det = fc.get_gif_t_a_w_det(gamma2, mu2, beta2, tauw2, taua2, delta2)
    prc_typ2 = gif_prc_theory(t_det2, a_det, w_det, gamma2, mu2, tauw2, beta2, taua2, delta2)


    st.set_default_plot_style()
    fig = plt.figure(tight_layout=True, figsize=(4, 2.5))
    grids = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(grids[0])
    ax2 = fig.add_subplot(grids[1], sharey=ax1)
    st.remove_top_right_axis([ax1, ax2])

    ax1.set_title("Type I")
    ax1.set_xlabel(r"$\tau/T^*$")
    #ax1.set_yticklabels([])
    ax1.set_ylabel(r"$Z(\tau)$")
    ax1.set_xlim([0, t_det1])
    ax1.set_xticks([0, t_det1/2, t_det1])
    ax1.set_xticklabels([0, 1/2, 1])
    ax1.plot([0, t_det1], [0,0], lw=1, c="C7")
    ax1.plot(np.linspace(0, t_det1, 100), prc_typ1, lw=1, color="k")
    ax1.fill_between(np.linspace(0, t_det1, 100), prc_typ1, 0, facecolor=st.colors[1], alpha=0.4, zorder=2)


    ax2.set_title("Type II")
    ax2.set_xlabel(r"$\tau/T^*$")
    #ax2.set_yticklabels([])
    ax2.set_ylabel(r"$Z(\tau)$")
    ax2.set_xlim([0, t_det2])
    ax2.set_xticks([0, t_det2/2, t_det2])
    ax2.set_xticklabels([0, 1/2, 1])

    ax2.plot([0, t_det2], [0,0], lw=1, c="C7")
    ax2.plot(np.linspace(0, t_det2, 100), prc_typ2, lw=1, color="k")
    prc_neg = [x for x in prc_typ2 if x < 0]
    prc_pos = [x for x in prc_typ2 if x >= 0]
    t0_step = len(prc_neg)
    t0 = t0_step / 100 * t_det2
    ax2.fill_between(np.linspace(0, t0, t0_step), prc_neg, 0, facecolor="C3", alpha=0.5, zorder=2)
    ax2.fill_between(np.linspace(t0, t_det2, 100 - t0_step), prc_pos, 0, facecolor=st.colors[1], alpha=0.5, zorder=2)

    home = os.path.expanduser("~")
    plt.savefig(home + "/Desktop/Presentations/SCC SfB/fig5.png", dpi=300)
    plt.show()