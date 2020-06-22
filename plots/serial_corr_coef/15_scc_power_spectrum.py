import numpy as np
import matplotlib.pyplot as plt
from utilites import functions as fc
from utilites import plot_parameters as utl
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import integrate
import cmath
import os


def gif_varprc(t, t_det, w_det, a_det, gamma,  mu, tau_a, delta, tau_gif, beta_gif):
    nu = 1. / tau_gif + gamma
    om02 = (beta_gif + gamma) / tau_gif
    A = nu ** 2 / 4 - om02
    if tau_a==0:
        ztt = 1./(mu - gamma * 1 - beta_gif*w_det)
    else:
        ztt = 1. / (mu - gamma * 1 - beta_gif * w_det - a_det + delta/tau_a)
    if A < 0:
        Om = np.sqrt(abs(A))
        return ztt * np.exp(0.5 * nu * (t - t_det)) * (
                np.cos(Om * (t - t_det)) + (gamma - 1. / tau_gif) / (2 * Om) * np.sin(Om * (t - t_det)))
    else:
        Om = np.sqrt(abs(A))
        return ztt * np.exp(0.5 * nu * (t - t_det)) * (
                np.cosh(Om * (t - t_det)) + (gamma - 1. / tau_gif) / (2 * Om) * np.sinh(Om * (t - t_det)))


def GIF_scc(t_det, w_det, gamma, mu, tau_gif, beta_gif, tau_a, delta, tau_n, Dn, Dw, k):
    if tau_a == 0:
        alpha = 0
        a_det = 0
        nu = 1
        def pa(i): return 0
    else:
        alpha = np.exp(-t_det / tau_a)
        a_det = delta / (tau_a*(1. - alpha))

        def h(t):
            return gif_varprc(t, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-t / tau_a)
        zi = integrate.quad(h, 0, t_det)[0]

        ts = np.linspace(0, t_det, 100)
        dt = t_det / 100
        zi = 0
        for t in ts:
            zi+= gif_varprc(t, t_det, w_det, a_det, gamma,mu, tau_a, delta, tau_gif, beta_gif)* np.exp(-t / tau_a)*dt

        nu = 1. - (a_det / tau_a) * zi

        def a(i):
            return alpha * (1 - alpha * alpha * nu) * (1 - nu) * np.real(cmath.exp((i - 1) * cmath.log(alpha * nu)))

        c = np.power(alpha, 2) - 2 * np.power(alpha, 2) * nu + 1

        def pa(i):
            return -a(i) / c

    if tau_n == 0:
        beta = 0
        chi2 = 0
        chi1 = 1
    else:
        beta = np.exp(-t_det / tau_n)
        chi2 = 0
        chi1 = 0
        ts = np.linspace(0, t_det, 100)
        dt = t_det / 100
        for t1 in ts:
            for t2 in ts:
                chi2 += gif_varprc(t1, t_det, w_det, a_det, gamma,mu, tau_a, delta, tau_gif, beta_gif) * gif_varprc(t2, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-abs(t_det + t2 - t1) / tau_n) * dt * dt
                chi1 += gif_varprc(t1, t_det, w_det, a_det, gamma,mu, tau_a, delta, tau_gif, beta_gif) * gif_varprc(t2, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) * np.exp(-abs(t2 - t1) / tau_n) * dt * dt
        for t1 in ts:
            chi1 += (2 * tau_n * Dw / Dn) * gif_varprc(t1, t_det, w_det, a_det, gamma, mu, tau_a, delta, tau_gif, beta_gif) ** 2 * dt

    def pn(i):
        return beta ** (i - 1) * chi2 / chi1

    A = (1. + (alpha * nu) ** 2 - 2 * alpha * nu * beta) * pn(1)
    B = 1 - alpha * nu * beta
    D = (1. - (alpha * nu) ** 2) * (1. - alpha * beta) / (1 + alpha ** 2 - 2 * alpha ** 2 * nu)
    C = 1 + 2 * pa(1) * pn(1) - alpha * nu * beta

    factor1 = (A / (alpha * nu - beta) + B) / C
    factor2 = (D / C) * (alpha - beta) / (alpha * nu - beta)
    return factor1 * pa(k) + factor2 * pn(k)


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
    vt =[]
    wt = []
    at = []
    tt = []
    N = 0
    while abs(a_tmp0 - a_tmp1) > 0.0001 and N <= 10:
        if v < 1.:
            v_tmp = v
            v += (mu - gamma*v - beta * w - a) * dt
            w += ((v_tmp - w) / tau_w) * dt
            a += (-a / tau_a) * dt
            t += dt
            t1 += dt
            tt.append(t1)
            vt.append(v)
            wt.append(w)
            at.append(a)
        else:
            print("FIRE!")
            t_det = t
            w_det = w
            a_tmp1 = a_tmp0
            a_tmp0 = a
            v = 0
            w = w0
            a += delta/tau_a
            t = 0
            N +=1
    return t_det, w_det, a, tt, vt, wt, at


def gif_prc_theory(t_det, a_det, w_det, gamma, mu, tau_w, beta, tau_a, delta):
    prc = []
    nu = gamma + 1/tau_w
    w = np.sqrt((beta+gamma)/tau_w - nu**2/4)
    print( ((1.-tau_w*gamma)/(2*tau_w*w)))
    for t in np.linspace(0, t_det, 100):
        varprc = np.exp((nu/2)*(t-t_det))*(np.cos(w*(t-t_det)) - ((1.-tau_w*gamma)/(2*tau_w*w))*np.sin(w*(t-t_det)))/(mu-gamma-beta*w_det - a_det + delta/tau_a)
        prc.append(varprc)
    return prc, w


def plot_prc():
    mu = 10
    gamma = 1
    betaw = 3
    tauw = 1.5
    taua = 10
    delta = 10
    t_det, w_det, a_det = get_gif_t_a_w_det(gamma, mu, betaw, tauw, taua, delta)
    prc, omega = gif_prc_theory(t_det, a_det, w_det, gamma, mu, tauw, betaw, taua, delta)

    f = plt.figure(1, figsize=utl.adjust_plotsize(1.))
    x0 = 0.12
    x1 = 0.05
    y0 = 0.4
    y1 = 0.10
    height = (1 - y0 - y1)
    width = (1 - x0 - x1)

    ax = f.add_axes([x0, y0, width, height])
    l, = ax.plot(np.linspace(0, t_det, 100), prc)
    ax.set_xticks([0, 0.5*t_det, t_det])
    ax.set_xticklabels([0, 1/2, 1])

    axbetaw = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="w")
    axtauw = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="w")
    axmu = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor="w")
    axtaua = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor="w")
    axdelta = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor="w")
    t1 = plt.text(0.1, 0.9, r"$T^*:  {:.2f}$".format(t_det), transform=ax.transAxes)
    t2 = plt.text(0.1, 0.85, "$\Omega: {:.2f}$".format(omega), transform=ax.transAxes)
    #t3 = plt.text(0.1, 0.80, ((1.-tau_w)/(2*tau_w*omega)), transform=ax.transAxes)

    gamma0 = 1.
    beta0 = 3
    tau_w0 = 1.5
    mu0=10
    taua=10
    delta=10
    #sgamma = Slider(axgamma , r"$\gamma$", 0, 10., valinit=gamma0, valstep=0.1)
    sbetaw = Slider(axbetaw, r"$\beta$", 0.0, 10.0, valinit=beta0, valstep=0.1)
    stauw = Slider(axtauw, r"$\tau_w$", 0.0, 10.0, valinit=tau_w0, valstep=0.1)
    smu = Slider(axmu, r"$\mu$", 1.75, 20.0, valinit=mu0, valstep=0.1)
    staua = Slider(axtaua, r"$\tau_a$", 0, 20.0, valinit=mu0, valstep=0.5)
    sdelta = Slider(axdelta, r"$\Delta$", 0, 20.0, valinit=mu0, valstep=1)

    def update(val):
        betaw = sbetaw.val
        tauw = stauw.val
        mu = smu.val
        taua= staua.val
        delta =sdelta.val
        t_det, w_det, a_det = get_gif_t_a_w_det(gamma, mu, betaw, tauw, taua, delta)
        prc, omega = gif_prc_theory(t_det, a_det, w_det, gamma, mu, tauw, betaw, taua, delta)
        l.set_ydata(prc)
        f.canvas.draw_idle()
        t1.set_text(r"$T^*:  {:.2f}$".format(t_det))

    sbetaw.on_changed(update)
    stauw.on_changed(update)
    smu.on_changed(update)
    sdelta.on_changed(update)
    staua.on_changed(update)
    plt.show()
    return 1


def plot_single_prc():
    for i in range(1):
        mu = 1.0 #1.5
        gamma = -1.0
        betaw = 5 #1.5
        tauw = -1/gamma + 0.1
        taua = 1.0
        delta = 2.3*taua #0
        w0 = 0 #1
        t_det, w_det, a_det, tt, vt, wt, at = get_gif_t_a_w_det(w0, gamma, mu, betaw, tauw, taua, delta)
        f1 = plt.figure(1, figsize=utl.adjust_plotsize(1.))
        x0 = 0.12
        x1 = 0.05
        y0 = 0.4
        y1 = 0.10
        height = (1 - y0 - y1)
        width = (1 - x0 - x1)
        print("t_det", t_det)

        ax4 = f1.add_axes([x0 + (width-0.1)/2 + 0.1, 0.7, (width-0.1)/ 2, 0.25])
        ax3 = f1.add_axes([x0, 0.7, (width-0.1)/2, 0.25])
        ax2 = f1.add_axes([x0, 0.4, width, 0.25])
        ax1 = f1.add_axes([x0, 0.1, width, 0.25])
        ax1.plot(tt, vt)
        ax2.plot(tt, wt)

        prc, omega = gif_prc_theory(t_det, a_det, w_det, gamma, mu, tauw, betaw, taua, delta)
        ts = np.linspace(0, t_det, 100)
        ax3.plot(ts, prc)

        prca = [p*(a_det/taua)*np.exp(-t/taua) for p, t in zip(prc, ts)]
        sum = 0
        for p in prca:
            sum += p
        print(sum)
        ax3.plot(ts, prca)

        scc= []
        sigma = 0.01
        taun = 10*t_det

        for k in range(1,6):
            scc.append(GIF_scc(t_det, w_det, gamma, mu, tauw, betaw, taua, delta, taun, 0.1, 0, k))
        ax4.plot([k for k in range(1,6)] , scc)
        plt.show()

if __name__ == "__main__":
    plot_single_prc()
    #t_det = plot_prc()
