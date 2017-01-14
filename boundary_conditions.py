from __future__ import print_function
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()

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

def test_dirichlet_bcs():
    divs = [[20,20], [20,20, 20]]
    L = [[-1., -1, 2., 1.], [-1., -1, 0, 2., 1., 1.]]
    mesh_type = ["UnitHyperCube", "HyperCube"]
    tol = 1E-10
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    f = Constant(-6.0)

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")
            if i == 0:
                mesh = UnitHyperCube(divisions)
            elif i == 1:
                mesh = HyperCube(L[j], divisions)
            bc, V, VV, V_g = dirichlet_bcs(u_D, mesh)
            phi = dirichlet_solver(f, V, bc)

            # error_l2 = errornorm(u_D, phi, "L2")
            # print("l2 norm: ", error_l2)

            vertex_values_u_D = u_D.compute_vertex_values(mesh)
            vertex_values_phi = phi.compute_vertex_values(mesh)

            error_max = np.max(vertex_values_u_D - \
                                vertex_values_phi)

            msg = 'error_max = %g' %error_max
            assert error_max < tol , msg

            plot(phi, interactive=True)
            plot(u_D, mesh=mesh, interactive=True)

            # The gradient:
            #flux_u_x, flux_u_y = E.split(deepcopy=True)

            # Exact flux expressions
            # u_e = lambda x, y: 1 + x**2 + 2*y**2
            # flux_x_exact = lambda x, y: -2*x
            # flux_y_exact = lambda x, y: -4*y
            #
            # # Compute error in flux
            # coor = phi.function_space().mesh().coordinates()
            # for i, value in enumerate(flux_u_x.compute_vertex_values()):
            #     print('vertex %d, x = %s, -p*u_x = %g, error = %g' %
            #           (i, tuple(coor[i]), value, flux_x_exact(*coor[i])))
            # for i, value in enumerate(flux_u_y.compute_vertex_values()):
            #     print('vertex %d, x = %s, -p*u_y = %g, error = %g' %
            #           (i, tuple(coor[i]), value, flux_y_exact(*coor[i])))

def test_periodic_bcs():
    divs = [[400, 400], [10, 10, 10]]
    l_unit = [[0., 0., 1., 1.], [0., 0., 0., 1., 1., 1.]]
    l_hyper = [[-1., -1, 1., 1.],[-1., -1., -1., 1., 1., 1.]]

    mesh_type = ["UnitHyperCube", "HyperCube"]
    tol = 1E-6

    for i in range(len(mesh_type)):
        print("------ Test of mesh type ", mesh_type[i], "   ------")
        for j in range(len(divs)):
            divisions = divs[j]
            print(len(divisions), "D test with ", divisions, " nodes.")
            if i == 0:
                L = l_unit[j]
                mesh = UnitHyperCube(divisions)
            if i == 1:
                L = l_hyper[j]
                mesh = HyperCube(L, divisions)

            V, VV, V_g = periodic_bcs(mesh, L)

            if j == 0:
                class Source(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*(8.0*DOLFIN_PI*DOLFIN_PI)

                class Exact(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])

            if j == 1:
                class Source(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*cos(2.0*DOLFIN_PI*x[2])*(12.0*DOLFIN_PI*DOLFIN_PI)

                class Exact(Expression):
                    def eval(self, values, x):
                        values[0] = sin(2.0*DOLFIN_PI*x[0])*sin(2.0*DOLFIN_PI*x[1])*cos(2.0*DOLFIN_PI*x[2])

            f = Source(degree=2)
            phi_e = Exact(degree=2)
            phi = periodic_solver(f, V)

            #error_l2 = errornorm(phi_e, phi, "L2")
            #print("l2 norm: ", error_l2)

            vertex_values_phi_e = phi_e.compute_vertex_values(mesh)
            vertex_values_phi = phi.compute_vertex_values(mesh)

            error_max = np.max(vertex_values_phi_e - \
                                vertex_values_phi)

            msg = 'error_max = %g' %error_max
            print("error_max: ", error_max)
            #assert error_max < tol , msg

            #plot(phi, interactive=True)
            #plot(phi_e, mesh=mesh, interactive=True)

if __name__=='__main__':

    from mesh_types import *
    from FieldSolver import *
    import numpy as np

    # Run the tests
    test_periodic_bcs()
    # test_dirichlet_bcs()
