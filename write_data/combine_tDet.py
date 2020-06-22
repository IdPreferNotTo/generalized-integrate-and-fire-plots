import numpy as np
import os

def get_mu(data_file):
    with open(data_file, "r") as data:
        for i, line in enumerate(data):
            if i == 0:
                mu_str = line
            elif i >= 1:
                break
    mu_splitted_str = mu_str.split()
    mu = float(mu_splitted_str[4][:-1])
    return mu


def combine_tDet():
    home = os.path.expanduser('~')
    final_file = "adap_LIF_tDet_gamma1.0_vT0.1.dat"
    final_data = []
    with open(home + "/Data/cLIF/" + final_file, "w") as f:
        for file in os.listdir(home + "/Data/cLIF/"):
            if file.endswith("0.1"):
                mu = get_mu(home + "/Data/cLIF/" + file)
                print(mu)
                data = np.loadtxt(home + "/Data/cLIF/" +file)
                for set in data:
                    final_data.append([mu, set[0], set[1], set[2]])
        final_data.sort(key=lambda x: x[0])
        f.write("# mu tau_a delta T \n")
        for data in final_data:
            f.write("{:.2f} {:.1f} {:.1f} {:.5f}\n".format(*data))
    return 0

if __name__ == "__main__":
    combine_tDet()