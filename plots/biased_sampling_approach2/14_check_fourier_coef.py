import numpy as np
import matplotlib.pyplot as plt

def plt_fourier_series():
    f, ax = plt.subplots(1, 1, figsize=(3, 9 / 4))
    ax.tick_params(direction='in')

    mu = 1.01
    vR = 0
    vT = 1
    t_det = np.log((mu - vR) / (mu - vT))
    f_vT = mu - vT
    cn = lambda n : np.power(t_det*f_vT, -1)*(1 -np.exp(-t_det))/(1 - 1j*2*np.pi*n/t_det)

    ts =  np.linspace(0, 3*t_det+0.05, 100)
    y2 = []
    for t in ts:
        y_fourier = 0
        for n in range(-100, 100):
            y_fourier += cn(n)*np.exp(1j*2*np.pi*n*t/t_det)
        y2.append(np.real(y_fourier))
    y = np.exp((ts%t_det)-t_det)/(mu-vT)
    ax.plot(ts,y)
    ax.plot(ts, y2)
    plt.show()
    return 1


if __name__ == "__main__":
    plt_fourier_series()