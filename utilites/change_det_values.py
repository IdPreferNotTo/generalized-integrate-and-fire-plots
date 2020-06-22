import numpy as np
import fileinput
import os

def change_det_values(mu, tau_a, tau_n, D, delta, ratio):
    home = os.path.expanduser('~')
    data_file = home + "/Data/GIF/white/data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.5e}_Delta{:.1f}_0.txt".format(mu, tau_a, tau_n, D, delta)
    replace_file = home + "/Data/GIF/white/replace_data/mu{:.2f}_tau_a{:.1f}_tau_n{:.1f}_D{:.5e}_Delta{:.1f}_ratio{:.2f}.txt".format(mu, tau_a, 0, D, delta, 0)

    rep_data = []
    with open(replace_file, "r") as rep_data_tmp:
        for n, line in enumerate(rep_data_tmp):
            if n<=5:
                line = line.rstrip('\n')
                rep_data.append(line)
            else:
                break

    for n, line in enumerate(fileinput.input(data_file, inplace=True)):
        if n <= 5:
            print(rep_data[n])
        elif(n>=100):
            line = line.rstrip('\n')
            print(line)
    return 1


if __name__ == "__main__":
    for i in range(5):
        for k in range(50):
            mu = [10, 11.75, 20, 2.12, 1.5][i]
            tau_a = [10, 10, 10, 1, 1][i]
            gamma = 1
            delta = [1, 1, 1, 10, 9][i]
            beta = [3, 3, 1.5, 1.5, 1.5][i]
            tau_w = 1.5
            D = [10 ** (-4), 10 ** (-4), 10 ** (-4), 10 ** (-4), 10 ** (-5)][i]
            ratio = 0
            tau_n = 0.2*k
            change_det_values(mu, tau_a, tau_n, D, delta, ratio)