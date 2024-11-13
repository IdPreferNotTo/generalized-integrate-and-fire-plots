import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
import os

def plt_prc_gif(gamma, mu, tau_a, tau_w, beta, D, delta):

    home = os.path.expanduser('~')
    data_file = home + "/Data/GIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
        mu, tau_a, 0, D, delta, 0)
    t_det = fc.read_t_det(data_file)
    a_det = fc.read_a_det(data_file)
    w_det = fc.read_w_det(data_file)
    ts = np.linspace(0, t_det, 100)
    data = np.loadtxt(data_file)
    t, a, eta, chi, chi2 = np.transpose(data)

    gif_prc = []
    for t in ts:
        gif_prc.append(fc.prc_gifadap(t,t_det,w_det,a_det,mu,gamma,beta,tau_w,delta)*np.exp(-t/tau_a))
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.plot(ts, gif_prc)
    plt.show()

    return 1

def plt_prc_lif(gamma, mu, tau_a, delta):
    home = os.path.expanduser('~')
    data_file = home + "/Data/LIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.2f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(
        mu, tau_a, 0.02, D, delta, 0)
    t_det = fc.read_t_det(data_file)
    ts = np.linspace(0, t_det, 100)
    lif_prc = []
    for t in ts:
        lif_prc.append(fc.lif_varprc(t, t_det, gamma, tau_a, delta, ))
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.plot(ts, lif_prc)
    plt.show()
    return 1

if __name__ ==  "__main__":
    for j in range(0):
        gamma = 1.
        mu = [5, 20, 20][j]
        v_t = 1
        tau_a = 2
        delta = [1, 4.47, 10][j]
        D = 0.1
        plt_prc_lif(gamma, mu, tau_a, delta)

    for j in range(5):
        gamma = 1
        mu = [10, 11.75, 20, 2.12, 1.5][j]
        tau_a = [10, 10, 10, 1, 1][j]
        tau_w = 1.5
        beta = [3, 3, 1.5, 1.5, 1.5][j]
        delta = [1, 1, 1, 10, 9][j]
        D = [1e-4, 1e-4, 1e-4, 1e-4, 1e-5][j]
        plt_prc_gif(gamma, mu, tau_a, tau_w, beta, D, delta)