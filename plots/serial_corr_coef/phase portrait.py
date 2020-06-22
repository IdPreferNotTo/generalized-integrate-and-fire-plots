import numpy as np
import matplotlib.pyplot as plt
from utilites import plot_parameters as utl
import os


def f(XY, gamma, mu, beta, tauw):
    x, y = XY
    return [mu - gamma*x - beta*y, (x-y)/tauw]


x = np.linspace(-2.0, 2.0, 20)
y = np.linspace(-2.0, 2.0, 20)

X, Y = np.meshgrid(x, y)

t = 0

u, v = np.zeros(X.shape), np.zeros(Y.shape)

NI, NJ = X.shape

gamma = -1.0
mu = 1.
beta = 5
tauw = 1.1

for i in range(NI):
    for j in range(NJ):
        x = X[i, j]
        y = Y[i, j]
        yprime = f([x, y], gamma, mu, beta, tauw)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

v0 = 0
w0 = 1
vt = []
wt = []
dt = 0.0001
while v0 < 1:
    vt.append(v0)
    wt.append(w0)
    vtmp = v0
    wtmp = w0
    v0 += (mu - gamma*vtmp - beta*wtmp)*dt
    w0 += ((vtmp-wtmp)/tauw) * dt


f = plt.figure(1, figsize=utl.adjust_plotsize(0.7))
x0 = 0.2
x1 = 0.05
y0 = 0.2
y1 = 0.05
height = (1 - y0 - y1)
width = (1 - x0 - x1)
ax = f.add_axes([x0, y0, width, height])


Q = ax.quiver(X, Y, u, v, color='k')
ax.scatter(mu/(gamma+beta), mu/(gamma+beta), c="w", edgecolors="k")
ax.vlines(1, ymin=-5, ymax=5)
ax.vlines(0, ymin=-5, ymax=5, ls="--")

ax.plot(vt, wt, c="C3")
ax.scatter(0, 1, c="C3", edgecolors="k",zorder=3)
ax.set_xlabel('$v$')
ax.set_ylabel('$w$')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
home = os.path.expanduser("~")
plt.savefig(home + "/Data/Plots_paper/phase_portrait_gif.pdf", transparent=True)
plt.show()


