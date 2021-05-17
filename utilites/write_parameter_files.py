import json
import numpy as np
import os


def write_parameter_file(N, model, gamma, mu, beta_gif, tau_gif, reset_gif, tau_a, jump_a, tau_n, D_n, D_w):
    home = os.path.expanduser('~')
    with open(home + '/Parameter/{}.json'.format(N), 'w') as file:
        json.dump({
            "num parameter": {
                "run": N,
                "step size": 10 ** (-5),
                "max spikes": 100_000,
            },
            "neuron model": {
                "gamma" : gamma,
                "mu": mu,
                "model": model,
                "GIF": {
                    "beta": beta_gif,
                    "time constant": tau_gif,
                    "reset": reset_gif
                },
                "Theta": {
                    "beta": 1
                }
            },
            "adaptation": {
                "time constant": tau_a,
                "strength": jump_a,
            },
            "noise": {
                "intensity": D_w,
            },
            "OU": {
                "intensity": D_n,  # sigma^2 = D/tau_n => D = sigma^2 * tau_n
                "time constant": tau_n
            }
        },
            file, indent=0)


if __name__ == "__main__":
    N = 0
    for m in range(1):
        for i in range(1):
            for k in range(1):
                model = "LIF"
                if model=="GIF":
                    #t_det = 1.914680000004076
                    #gamma = -1
                    #mu = 1.
                    #beta_gif = 5
                    #tau_gif = 1.1
                    #reset_gif = 0 #1
                    #tau_a = 1
                    #jump_a = 2.3 #0
                    #tau_n = [2*t_det, 5*t_det][k] # 0.01 * t_det * (1.5 ** k)
                    #D_n = [0.01 * tau_n, 0.001 * tau_n, 0.0001 * tau_n][m]
                    #D_w = 0
                    #write_parameter_file(N, model, gamma, mu, beta_gif, tau_gif, reset_gif, tau_a, jump_a, tau_n, D_n, D_w)
                    #N += 1

                    t_det = [1.2351499999996243, 0.5670200000000543, 1.914680000004076][i]
                    gamma = [1, 1, -1][i]
                    mu = [10, 20, 1.][i]
                    beta_gif = [3, 1.5, 5][i]
                    tau_gif = [1.5, 1.5, 1.1][i]
                    tau_a = [10, 10, 1][i]
                    jump_a = [10, 10, 2.3][i]
                    tau_n = [0.1, 1, 10][i]
                    reset_gif = 0
                    print(0.0001* 10 ** (k/5))
                    D_n = 0.0001 * 10**(k/5)  #0.1
                    D_w = 0.01 #0.0001 * 10**(k/5)
                    write_parameter_file(N, model, gamma, mu, beta_gif, tau_gif, reset_gif, tau_a, jump_a, tau_n, D_n, D_w)
                    N+=1
                if model=="LIF":
                    gamma = 1
                    mu = 5.
                    beta_gif = 0
                    tau_gif = 0
                    reset_gif = 0
                    jump_a = 2
                    tau_n = 1.
                    tau_a = 2
                    D_n = 0.33/4     # sigma^2  = D/tau, thus if sigma is to be constant D must be increased propotional to0  tau
                    D_w = 0.50/4
                    write_parameter_file(N, model, gamma, mu, beta_gif, tau_gif, reset_gif, tau_a, jump_a, tau_n, D_n, D_w)
                    N+=1
