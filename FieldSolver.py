# Poisson solver with periodic and Dirichlet boundary conditions
#
#     - div grad u(x, y, z) = f(x, y, z)
#
from __future__ import print_function
import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()

def periodic_solver(f, V):

    # Create Krylov solver
    solver = PETScKrylovSolver('cg', 'hypre_amg')#'gmres', 'ilu')

    solver.parameters["absolute_tolerance"] = 1e-14
    solver.parameters["relative_tolerance"] = 1e-12
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["monitor_convergence"] = True
    solver.parameters["convergence_norm_type"] = "true"
    for item in solver.parameters.items(): print(item)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    A, b = assemble_system(a, L)
    phi = Function(V)

    solver.set_operator(A)

    # Create vector that spans the null space and normalize
    null_vec = Vector(phi.vector())
    V.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")

    # Create null space basis object and attach to PETSc matrix
    null_space = VectorSpaceBasis([null_vec])
    as_backend_type(A).set_nullspace(null_space)

    # Orthogonalize RHS vector b with respect to the null space
    null_space.orthogonalize(b);

    # Solve
    solver.solve(phi.vector(), b)
    return phi

def dirichlet_solver(f, V, bc):

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    phi = Function(V)
    solve(a == L, phi, bc)
    return phi

def E_field(phi, VV):
    # Compute gradient
    V = phi.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'CG', degree)

    grad_phi = Function(VV)
    grad_phi = project(-1*grad(phi), VV)
    e_field = project(grad_phi, W)
    return e_field

def test_periodic_solver():
    divs = [[100, 100]]#, [150, 100], [100, 150], [200,200]]
    L = [[-1., 0., 1., 1.], [-1., -1, 0, 2., 1., 1.]]
    tol = 1E-9

    for i in range(len(divs)):
        print("run: ", i)
        divisions = divs[i]
        print('divisions: ', divisions)
        #mesh = UnitHyperCube(divisions)
        mesh = HyperCube(L[0], divisions)
        V, V_g = periodic_bcs(mesh, L[0])

        class Source(Expression):
            def eval(self, values, x):
                values[0] = sin(2.0*DOLFIN_PI*x[0]) + sin(2.0*DOLFIN_PI*x[1])

        class Exact(Expression):
            def eval(self, values, x):
                values[0] = (sin(2.0*DOLFIN_PI*x[0]) + sin(2.0*DOLFIN_PI*x[1]))/\
                            (4.0*DOLFIN_PI*DOLFIN_PI)

        # class Source(Expression):
        #     def eval(self, values, x):
        #         values[0] = sin(2.0*DOLFIN_PI*x[0]) + sin(6.0*DOLFIN_PI*x[0])
        #
        # class Exact(Expression):
        #     def eval(self, values, x):
        #         values[0] = sin(2.0*DOLFIN_PI*x[0])/(4.0*DOLFIN_PI*DOLFIN_PI) +\
        #                     sin(6.0*DOLFIN_PI*x[0])/(36.0*DOLFIN_PI*DOLFIN_PI)


        f = Source(degree=4)
        phi_e = Exact(degree=4)

        phi = periodic_solver(f, V)

        error_l2 = errornorm(phi_e, phi, "L2")
        print("l2 norm: ", error_l2)

        vertex_values_phi_e = phi_e.compute_vertex_values(mesh)
        vertex_values_phi = phi.compute_vertex_values(mesh)

        error_max = np.max(vertex_values_phi_e - \
                            vertex_values_phi)

        msg = 'error_max = %g' %error_max
        assert error_max < tol , msg

        e_field = E_field(phi, V_g)
        # e_x = Expression('-2*x[0]', degree=1)
        # e_y = Expression('-4*x[1]', degree=1)
        # # The gradient:
        # flux_u_x, flux_u_y = E.split(deepcopy=True)
        #
        plot(phi, interactive=True)
        plot(e_field, interactive=True)

if __name__ == '__main__':
    import time
    from mesh_types import *
    from boundary_conditions import *

    def run_periodic_solver():
        L = [-1., 1.,-1., 1.]
        # The mesh
        #mesh = UnitSquareMesh(10,10)#RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
        mesh = RectangleMesh(Point(L[0],L[2]), Point(L[1],L[3]), 20, 20)
        class Source(Expression):
            def eval(self, values, x):
                values[0] = sin(2.0*DOLFIN_PI*x[0]) #+ sin(2.0*DOLFIN_PI*x[1])

        f = Source(degree=2)
        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(SubDomain):

            def __init__(self, L):
                dolfin.SubDomain.__init__(self)
                self.Lx_left = L[0]
                self.Lx_right = L[1]
                self.Ly_left = L[2]
                self.Ly_right = L[3]
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
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
                else:   # near(x[1], 1)
                    y[0] = x[0]
                    y[1] = x[1] - (self.Ly_right - self.Ly_left)

        # Create boundary and finite element
        PBC = PeriodicBoundary(L)
        V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
        V_g = VectorFunctionSpace(mesh, 'DG', 0, constrained_domain=PBC)

        #ff = project(f, V)
        phi, E = periodic_solver(f, mesh, V, V_g)

        #plot(ff,  interactive=True)
        plot(phi, interactive=True)
        plot(E, interactive=True)
        # grad_u_x, grad_u_y = E.split(deepcopy=True)  # extract components
        #
        # u_array = phi.vector().array()
        # grad_u_x_array = grad_u_x.vector().array()
        # grad_u_y_array = grad_u_y.vector().array()
        # coor = mesh.coordinates()
        # for i in range(len(u_array)):
        #     x, y = coor[i]
        #     print 'Node (%.3f,%.3f): u = %.4f (%9.2e), '\
        #           'grad(u)_x = %.4f  (%9.2e), grad(u)_y = %.4f  (%9.2e)' % \
        #           (x, y, u_array[i], (-1.*sin(2.0*DOLFIN_PI*x))/(4.0*DOLFIN_PI*DOLFIN_PI) - u_array[i],
        #            grad_u_x_array[i], cos(2.0*DOLFIN_PI*x)/(2.0*DOLFIN_PI) - grad_u_x_array[i],
        #            grad_u_y_array[i], 0 - grad_u_y_array[i])
        #           (x, y, u_array[i], (sin(4.0*DOLFIN_PI*x) + sin(4.0*DOLFIN_PI*y))/(16.0*DOLFIN_PI*DOLFIN_PI) - u_array[i],
        #            grad_u_x_array[i], cos(4.0*DOLFIN_PI*x)/(4.0*DOLFIN_PI) - grad_u_x_array[i],
        #            grad_u_y_array[i], cos(4.0*DOLFIN_PI*y)/(4.0*DOLFIN_PI) - grad_u_y_array[i])

    def run_Dirichlet_solver():
        # Source term
        class Source(Expression):
            def eval(self, values, x):
                dx = x[0] - 0.5
                dy = x[1] - 0.5
                values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) \
                            + 1.0*exp(-(dx*dx + dy*dy)/0.02)

        class Source1(Expression):
            def eval(self, values, x):
                values[0] = x[0]*exp(-pow(pow(2*(x[0]-0.5),2)+pow(2*(x[1]-0.5),2),2))

        # Create mesh and finite element
        mesh = UnitSquareMesh(20, 20)
        V = FunctionSpace(mesh, "CG", 1)
        V_g = VectorFunctionSpace(mesh, 'CG', 1)

        # Create Dirichlet boundary condition
        u0 = Constant(0.0)
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u0, boundary)

        f = Source1(degree=1)
        phi, E = dirichlet_solver(f, V, V_g, bc)
        plot(phi, interactive=True)
        plot(E, interactive=True)
        #File("data/phi.pvd") << phi
        #sFile("data/partitions.pvd") << CellFunction("size_t", mesh, rank)

    # tic = time.time()
    #run_Dirichlet_solver()
    # toc = time.time()
    # if rank == 0:
    #     seconds = toc-tic
    #     print "Total run time: %d seconds " %seconds
    #     m, s = divmod(seconds, 60)
    #     h, m = divmod(m, 60)
    #     print "Total run time: %d:%02d:%02d" % (h, m, s)

    test_periodic_solver()
