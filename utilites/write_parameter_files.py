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
        for i in range(3):
            for k in range(1):
                model = "LIF"
                if model=="GIF":
                    gamma = 1
                    mu = 20
                    beta_gif = 1.5
                    tau_gif = 1.5
                    tau_a = 10
                    jump_a = 0
                    tau_n = 1
                    reset_gif = 0
                    D_n = 0.0001 * 10**(k/5)
                    D_w = 0.01
                    write_parameter_file(N, model, gamma, mu, beta_gif, tau_gif, reset_gif, tau_a, jump_a, tau_n, D_n, D_w)
                    N+=1
                if model=="LIF":
                    gamma = 1
                    mu = 5
                    beta_gif = 0
                    tau_gif = 0
                    reset_gif = 0
                    jump_a = 0
                    tau_n = [0.02231, 0.2231, 2.231][i]
                    tau_a = 0
                    sigma2 = 0.02
                    D_n = sigma2*tau_n     # sigma^2  = D/tau, thus if sigma is to be constant D must be increased propotional to0  tau
                    D_w = 0.08
                    write_parameter_file(N, model, gamma, mu, beta_gif, tau_gif, reset_gif, tau_a, jump_a, tau_n, D_n, D_w)
                    N+=1
