import numpy as np
import matplotlib.pyplot as plt
import os

def plot_delta_ai():
    home = os.path.expanduser('~')
    data_file = home + "/Data/LIF/data/mu20.00_taua2.0_Delta10.0_taun0.02_Dn1.00e-01_Dw0.00e+00.txt"
    data = np.loadtxt(data_file)
    t, a, eta, chi = np.transpose(data)
    plt.hist(a, 100, density=True)
    var = np.var(a)
    gaus = []
    xs = np.linspace(-3*np.sqrt(var),3*np.sqrt(var), 100)
    for x in xs:
        gaus.append((1/np.sqrt(2*np.pi*var))*np.exp(-(x**2)/(2*var)))
    plt.plot(xs, gaus, c="k", lw=2)
    plt.yscale("log")
    plt.show()
    return 1

if __name__ == "__main__":
    plot_delta_ai()