# coding:utf-8

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
import time

class CartesianGrid:                                                            #基本構造
    """
        Simple class to generate a computational grid and apply boundary conditions
    """

    def __init__(self, nx=100, ny=100, nz=100, xmin=-5, xmax=5, ymin=-5, ymax=5, zmin=-5, zmax=5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ntotal = nx*ny*nz

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

        self.dx = (xmax - xmin)/(nx - 1)
        self.dy = (ymax - ymin)/(ny - 1)
        self.dz = (zmax - zmin)/(nz - 1)

        self.x = np.arange(xmin, xmax + 0.5*self.dx, self.dx)
        self.y = np.arange(ymin, ymax + 0.5*self.dy, self.dy)
        self.z = np.arange(zmin, zmax + 0.5*self.dz, self.dz)

    def create_field(self):
        return np.zeros((self.nx, self.ny), dtype=np.float)                     #条件を入れるための格納庫、配列

    def create_meshgrid(self):                                                  #軸設定、max, minで表示する範囲を決定
        return np.meshgrid(self.x, self.y, self.z)

    def set_boundary_condition1(self, V_1, r, eps=0.1):                                  #円柱状にデータを配列する
        a = r*r - V_1                                                             #rをmeshgridに合わせて考えること
        X ,Y = np.meshgrid(self.x, self.y)
        Z = X**2 + Y**2 - a
        for i in range(self.nx):
            for j in range(self.ny):
                if Z[i, j] > V_1 + eps or Z[i, j] < V_1 - eps:
                    Z[i, j] = 0

        Z = Z.reshape(1, self.nx, self.ny)
        return Z

    def set_boundary_condition2(self, phi, V_0=1e-100):
        for i in range(self.nx):
            for j in range(self.ny):
                if phi[0, i, j] != 0.0:            #修正
                    phi[0, i, j] = V_0
        return phi

    def make_einzel_lens(self, Z_V, Z_0, z1, z2, z3, z4, z5, z6):
        Zeros = np.zeros((1, self.nx, self.ny))
        Z_new = np.zeros((1, self.nx, self.ny))
        for i in range(int(self.nz*z1/(self.zmax - self.zmin))):
            Z_new = np.vstack((Z_new, Zeros))

        for i in range(int(self.nz*z1/(self.zmax - self.zmin)), int(self.nz*z2/(self.zmax - self.zmin))):
            Z_new = np.vstack((Z_new, Z_0))

        for i in range(int(self.nz*z2/(self.zmax - self.zmin)), int(self.nz*z3/(self.zmax - self.zmin))):
            Z_new = np.vstack((Z_new, Zeros))

        for i in range(int(self.nz*z3/(self.zmax - self.zmin)), int(self.nz*z4/(self.zmax - self.zmin))):
            Z_new = np.vstack((Z_new, Z_V))

        for i in range(int(self.nz*z4/(self.zmax - self.zmin)), int(self.nz*z5/(self.zmax - self.zmin))):
            Z_new = np.vstack((Z_new, Zeros))

        for i in range(int(self.nz*z5/(self.zmax - self.zmin)), int(self.nz*z6/(self.zmax - self.zmin))):
            Z_new = np.vstack((Z_new, Z_0))

        for i in range(int(self.nz*z6/(self.zmax - self.zmin)), int(self.nz)-1):
            Z_new = np.vstack((Z_new, Zeros))

        return Z_new

    def convert_to_1d_array(self, x):
        return x.reshape(self.ntotal, 1)

    def convert_to_3d_array(self, x):
        return x.reshape(self.nx, self.ny, self.nz)

def calc_jacobi_matrix(mesh, Z):
    """
        Create sparse matrix for Jacobi method
    """

    A = lil_matrix((mesh.ntotal, mesh.ntotal))

    for i in range(1, mesh.nz-1):
        for j in range(1, mesh.ny-1):
            for k in range(1, mesh.nx-1):

                p = i*mesh.nz**2 + j*mesh.nx + k
                p_ip1 = (i+1)*mesh.nz**2 + j*mesh.nx + k
                p_im1 = (i-1)*mesh.nz**2 + j*mesh.nx + k
                p_jp1 = i*mesh.nz**2 + (j+1)*mesh.nx + k
                p_jm1 = i*mesh.nz**2 + (j-1)*mesh.nx + k
                p_kp1 = i*mesh.nz**2 + j*mesh.nx + (k+1)
                p_km1 = i*mesh.nz**2 + j*mesh.nx + (k-1)

                if Z[i, j, k] != 0:                                             #0の時はe-10などで近似して、なんとかする
                    A[p, p] = 1.0

                else:
                    A[p, p_ip1] = 1/6
                    A[p, p_im1] = 1/6
                    A[p, p_jp1] = 1/6
                    A[p, p_jm1] = 1/6
                    A[p, p_kp1] = 1/6
                    A[p, p_km1] = 1/6

    return A.tocsc()

class IterationControl:
    """
        Class to control iteration loop
    """

    def __init__(self, max_iter, info_interval, tolerance):
        self.max_iter = max_iter
        self.info_interval = info_interval
        self.tolerance = tolerance
        self.eps = 1.0
        self.iter = 0

    def loop(self):
        self.iter += 1
        self.output_info()

        if self.eps < self.tolerance:
            return False
        elif self.iter > self.max_iter:
            print("max iteration reached")
            return False
        else:
            return True

    def calc_epsilon(self, dx):
        self.eps = np.max(abs(dx))

    def output_info(self):
        if self.iter % self.info_interval == 0:
            print("iter = %d, eps = %.3e" % (self.iter, self.eps))


#main code
def solve_eq():

    mesh = CartesianGrid()

    # einzel lens boundary condition
    einzel_V = mesh.set_boundary_condition1(V_1 = 100, r = 4, eps=0.2)
    einzel_0 = mesh.set_boundary_condition2(einzel_V)
    einzel_V = mesh.set_boundary_condition1(V_1 = 100, r = 4, eps=0.2)

    Einzel_Lens = mesh.make_einzel_lens(einzel_V, einzel_0, z1=1, z2=3, z3=4, z4=6, z5=7, z6=9)

    A = calc_jacobi_matrix(mesh, Einzel_Lens)

    k = mesh.convert_to_1d_array(Einzel_Lens)

    iter_control = IterationControl(10000, 100, 1e-3)

    start_time = time.time()

    while iter_control.loop():
        k_new = A.dot(k)
        iter_control.calc_epsilon(k_new - k)
        k, k_new = k_new, k

    # reshape for surface plotting
    k = mesh.convert_to_3d_array(k)
    return k
