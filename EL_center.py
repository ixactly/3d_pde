# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import three_d_pde as pde
import pickle
from matplotlib import cm

with open('einzel_lens.binaryfile', 'rb') as lens:
    V = pickle.load(lens)

mesh = pde.CartesianGrid()
V_pre = V[:, int(mesh.ny/2), :]
two_d_V = V_pre.transpose()

x, y = np.meshgrid(mesh.z, mesh.x)
fig, ax = plt.subplots()
surf = ax.contourf(x, y, two_d_V, cmap=cm.coolwarm)
ax.set_ylabel('x')
ax.set_xlabel('y')

fig.colorbar(surf, shrink=0.5, aspect=5).set_label("V[V]")
plt.gca().set_aspect('equal', adjustable='box')
plt.title('einzel_lens')

plt.show()
