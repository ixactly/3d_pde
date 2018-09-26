# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import three_d_pde as pde
from mpl_toolkits.mplot3d import Axes3D
import pickle

with open('einzel_lens.binaryfile', 'rb') as lens:
    V = pickle.load(lens)

mesh = pde.CartesianGrid()

nx = mesh.nx
ny = mesh.ny
nz = mesh.nz

Y ,Z ,X = mesh.create_meshgrid()

Ex = np.empty((nz, nx, ny))
Ey = np.empty((nz, nx, ny))
Ez = np.empty((nz, nx, ny))

delta_x = (mesh.xmax - mesh.xmin)/nx
delta_y = (mesh.ymax - mesh.ymin)/ny
delta_z = (mesh.zmax - mesh.zmin)/nz

for i in range(ny-1):
    for j in range(nz-1):
        for k in range(nx-1):
            Ez[i, j, k] = -(V[i+1, j, k]-V[i, j, k])/delta_z
            Ex[i, j, k] = -(V[i, j+1, k]-V[i, j, k])/delta_x
            Ey[i, j, k] = -(V[i, j, k+1]-V[i, j, k])/delta_y


fig = plt.figure()
ax = Axes3D(fig)

ax.quiver(X, Y, Z, Ex, Ey, Ez, color='red')
ax.set_xlim([mesh.xmin, mesh.xmax])
ax.set_ylim([mesh.ymin, mesh.ymax])
ax.set_zlim([mesh.zmin, mesh.zmax])

plt.show()
