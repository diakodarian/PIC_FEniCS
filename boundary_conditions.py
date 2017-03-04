from __future__ import print_function
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()

def near(x, y, tol=1e-8):
    if abs(x-y)<tol:
        return True
    else:
        return False

def periodic_bcs(mesh, l):
    # Sub domain for 2d Periodic boundary condition
    class PeriodicBoundary2D(SubDomain):

        def __init__(self, L):
            dolfin.SubDomain.__init__(self)
            self.Lx_left = L[0]
            self.Lx_right = L[2]
            self.Ly_left = L[1]
            self.Ly_right = L[3]

        def inside(self, x, on_boundary):
            return bool((near(x[0], self.Lx_left) or near(x[1], self.Ly_left)) and
                   (not((near(x[0], self.Lx_left) and near(x[1], self.Ly_right)) or
                        (near(x[0], self.Lx_right) and near(x[1], self.Ly_left)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],  self.Lx_right) and near(x[1], self.Ly_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1] - (self.Ly_right - self.Ly_left)
            elif near(x[0],  self.Lx_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1]
            else:
                y[0] = x[0]
                y[1] = x[1] - (self.Ly_right - self.Ly_left)

    # Sub domain for 3d Periodic boundary condition
    class PeriodicBoundary3D(SubDomain):

        def __init__(self, L):
            dolfin.SubDomain.__init__(self)
            self.Lx_left = L[0]
            self.Lx_right = L[3]
            self.Ly_left = L[1]
            self.Ly_right = L[4]
            self.Lz_left = L[2]
            self.Lz_right = L[5]
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool( ( near(x[0], self.Lx_left) or near(x[1], self.Ly_left) or near(x[2], self.Lz_left) ) and
                   (not( near(x[0], self.Lx_right) and near(x[1], self.Ly_right) and near(x[2], self.Lz_right) )) and on_boundary)

        def map(self, x, y):
            if near(x[0],  self.Lx_right) and near(x[1], self.Ly_right) and near(x[2], self.Lz_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1] - (self.Ly_right - self.Ly_left)
                y[2] = x[2] - (self.Lz_right - self.Lz_left)
            elif near(x[0],  self.Lx_right) and near(x[1], self.Ly_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1] - (self.Ly_right - self.Ly_left)
                y[2] = x[2]
            elif near(x[1],  self.Ly_right) and near(x[2], self.Lz_right):
                y[0] = x[0]
                y[1] = x[1] - (self.Ly_right - self.Ly_left)
                y[2] = x[2] - (self.Lz_right - self.Lz_left)
            elif near(x[1],  self.Ly_right):
                y[0] = x[0]
                y[1] = x[1] - (self.Ly_right - self.Ly_left)
                y[2] = x[2]
            elif near(x[0],  self.Lx_right) and near(x[2], self.Lz_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1]
                y[2] = x[2] - (self.Ly_right - self.Ly_left)
            elif near(x[0],  self.Lx_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1]
                y[2] = x[2]
            else:
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - (self.Lz_right - self.Lz_left)

    # Create boundary and finite element
    geo_dim = mesh.geometry().dim()
    if geo_dim == 2:
        PBC = PeriodicBoundary2D(l)
    if geo_dim == 3:
        PBC = PeriodicBoundary3D(l)
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    VV = VectorFunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    W = VectorFunctionSpace(mesh, 'DG', 0, constrained_domain=PBC)
    return V, VV, W

def dirichlet_bcs(u_D, mesh, degree = 1):
    # Create dolfin function spaces
    V = FunctionSpace(mesh, "CG", 1)
    VV = VectorFunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, 'DG', 0)

    # Create Dirichlet boundary condition
    def boundary(x, on_boundary):
      return on_boundary

    bc = DirichletBC(V, u_D, boundary)
    return bc, V, VV, W

def dirichlet_bcs_zero_potential(V, facet_f, n_components):
    d = V.mesh().topology().dim()
    bcs_Dirichlet = []
    for i in range(2*d):
        bc0 = DirichletBC(V, Constant(0.0), facet_f, (n_components+i+1))
        bcs_Dirichlet.append(bc0)

    return bcs_Dirichlet

def dirichlet_bcs_B_field(E0, V, facet_f, n_components):
    d = V.mesh().topology().dim()
    if d == 2:
        phi_0 = 'x[0]*Ex + x[1]*Ey'
        phi_l = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
    if d == 3:
        phi_0 = 'x[0]*Ex + x[1]*Ey + x[2]*Ez'
        phi_l = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])


    bcs_Dirichlet = []
    for i in range(2*d):
        bc0 = DirichletBC(V, phi_l, facet_f, (n_components+i+1))
        bcs_Dirichlet.append(bc0)

    return bcs_Dirichlet
