import os
import numpy as np

def run_CLIF(gamma, mu, tau_n, D, tau_a, Delta):
    home = os.path.expanduser('~')
    prog = home + "/CLionProjects/sch13/cmake-build-release/isi correlations plots"
    parameters = " {:.1f} {:.2f} {:.1f} {:.1f} {:.1f} {:.1f}".format(gamma, mu, tau_n, D, tau_a, Delta)
    os.system(prog + parameters)


def run_detLIF(gamma, mu, tau_n, D, tau_a, Delta):
    home = os.path.expanduser('~')
    prog = home + "/detLIF_T"
    parameters = " {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(gamma, mu, tau_n, D, tau_a, Delta)
    os.system(prog + parameters)

def combine_stupid_files(gamma, mu, D, tau_a, Delta):
    #god im lazy....
    home = os.path.expanduser('~')
    with open(home + "/Data/LIF_T/firing_rate_tau_a{:.1f}_D{:.1f}_delta{:.1f}.txt".format(tau_a, D, Delta), 'w') as datafile:
        for file in os.listdir(home + "/Data/det_LIF"):
            mu, t = np.loadtxt(home + "/Data/det_LIF/" + file)
            datafile.write("{:.2f} {:.6f} \n".format(mu, t))

if __name__ == "__main__":
    mus = np.logspace(0.01, 2, num = 100)
    #for mu in mus[80:]:
    for i in range(1):
        gamma = 1
        D = 0.1
        tau_a = 10
        tau_n = 1
        Delta = 1
        mu = 5
        #run_detLIF(gamma, mu, tau_n, D, tau_a, Delta)
    #combine_stupid_files(gamma, mu, D, tau_a, Delta)
        run_CLIF(gamma, mu, tau_n, D, tau_a, Delta)
