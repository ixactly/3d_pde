# coding:utf-8

import numpy as np
import three_d_pde as pde
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from matplotlib import cm

with open('einzel_lens_vector.binaryfile', 'rb') as lens:
    V = pickle.load(lens)

mesh = pde.CartesianGrid()
V_pre = V[:, int(mesh.ny/2), :]
V = V_pre.transpose()

m = (40e-3)/(6.0e+23)
q = 1.6e-19
V_extract = 2000
H = 3e-10
z0 = mesh.zmin
y0 = input("初期位置r:")
y0 = int(y0)
vz0 = np.sqrt(2*q*V_extract/m)
vy0 = 0
t = 0


Ez = np.empty((mesh.nz-1, mesh.nx-1))
Ey = np.empty((mesh.nz-1, mesh.nx-1))

delta_y = (mesh.xmax - mesh.xmin)/(mesh.nx*1000)
delta_z = (mesh.zmax - mesh.zmin)/(mesh.nz*1000)

for i in range(mesh.nx -1):
    for j in range(mesh.nz -1):
        Ey[i, j] = (V[i+1, j] - V[i, j])/delta_y
        Ez[i, j] = -(V[i, j+1] - V[i, j])/delta_z

Az = Ez*(q/m)
Ay = Ey*(q/m)

def Runge_Kutta(x0, a, v, h):

    k1 = v
    k2 = v+a*h/2
    k3 = v+a*h/2
    k4 = v+a*h

    x = x0 + 1000*(k1+2*k2+2*k3+k4)*h/6
    return x

fig = plt.figure()

ims = []

while mesh.zmin<=z0<=mesh.zmax and mesh.ymin<=y0<=mesh.ymax:
    t += H

    az = Az[int((mesh.ny-1)*(mesh.ymax-y0)/(mesh.ymax-mesh.ymin)), int((mesh.nz-1)*(z0+mesh.zmax)/(mesh.zmax - mesh.zmin))]
    ay = Ay[int((mesh.ny-1)*(mesh.ymax-y0)/(mesh.ymax-mesh.ymin)), int((mesh.nz-1)*(z0+mesh.zmax)/(mesh.zmax - mesh.zmin))]

    vz0 += az*H
    vy0 += ay*H

    z0 = Runge_Kutta(z0, az, vz0, H)
    y0 = Runge_Kutta(y0, ay, vy0, H)

    im = plt.plot(z0, y0, "o", color="red")
    ims.append(im)

print(z0, y0, vz0, vy0, t)
plt.xlim([mesh.zmin, mesh.zmax])
plt.ylim([mesh.xmin, mesh.xmax])

ani = animation.ArtistAnimation(fig, ims, interval=1)

plt.show()