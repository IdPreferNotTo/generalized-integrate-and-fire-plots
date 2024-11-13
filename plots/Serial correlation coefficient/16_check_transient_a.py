import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/lukas/Data/LIF/data/mu5.00_taua100.0_Delta2.0.txt")
t, a = np.transpose(data)
plt.plot(t, a)
plt.show()